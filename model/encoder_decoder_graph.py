import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse
from toolz.sandbox import unzip
from torch.utils.data import Dataset
from torchvision import transforms
import more_itertools
import numpy as np

from common import torch_util, graph_attention_layer
from common.character_embedding import CharacterEmbedding
from common.context_query_attention import CosineCrossAttention
from common.feed_forward_block import FeedForwardLayer
from common.input_embedding import WordCharInputEmbedding, RandomInitialInputEmbedding
from common.problem_util import to_cuda, get_gpu_index
from common.util import create_sequence_node_link, PaddedList
from model.transformer_lm import TransformerModel, gelu, Transformer, PositionEmbedding


class EncoderDecoderModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 in_vocabulary_size,
                 n_filters,
                 kernel_size, padding,
                 hidden_size,
                 out_vocabulary_size,
                 dynamic_graph_n_layer,
                 graph_attention_n_head,
                 graph_itr,
                 leaky_alpha):
        super().__init__()
        self._graph_itr = graph_itr
        self.input_embedding = CharacterEmbedding(in_vocabulary_size, embedding_dim, n_filters,
                 kernel_size, padding)
        self.hidden_state_transform = nn.Linear(n_filters, hidden_size * graph_attention_n_head)
        self.dynamic_graph = graph_attention_layer.DynamicGraphAdjacentMatrix(input_dim=
                                                                              hidden_size * graph_attention_n_head,
                                                                              n_layer=dynamic_graph_n_layer)
        self.graph_attention = graph_attention_layer.MultiHeadGraphAttentionLayer(hidden_size * graph_attention_n_head,
                                                                                  hidden_size,
                                                                                  0.2,
                                                                                  leaky_alpha,
                                                                                  graph_attention_n_head)
        self.output = nn.Linear(hidden_size*graph_attention_n_head, out_vocabulary_size)

    def forward(self, encoder_sequence, initial_decoder_sequence, fixed_graph, ):
        output_begin_idx = encoder_sequence.shape[1]
        node_representation = self.input_embedding(torch.cat((encoder_sequence, initial_decoder_sequence), dim=1))
        del encoder_sequence, initial_decoder_sequence
        node_representation = self.hidden_state_transform(node_representation)
        for _ in range(self._graph_itr):
            dynamic_graph = self.dynamic_graph(node_representation)
            dynamic_graph = fixed_graph+dynamic_graph
            node_representation = node_representation + self.graph_attention(node_representation, dynamic_graph)
        o = self.output(node_representation[:, output_begin_idx:, :])
        return o


class EncoderDecoderModelWithPairInput(nn.Module):
    def __init__(self,
                 embedding_dim,
                 in_vocabulary_size,
                 n_filters,
                 kernel_size, padding,
                 hidden_size,
                 out_vocabulary_size,
                 dynamic_graph_n_layer,
                 graph_attention_n_head,
                 graph_itr,
                 leaky_alpha):
        super().__init__()
        self._graph_itr = graph_itr
        self.input_embedding = CharacterEmbedding(in_vocabulary_size, embedding_dim, n_filters,
                 kernel_size, padding)
        self.hidden_state_transform = nn.Linear(n_filters, hidden_size * graph_attention_n_head)
        dynamic_graph = graph_attention_layer.DynamicGraphAdjacentMatrix(input_dim=
                                                                         hidden_size * graph_attention_n_head,
                                                                         n_layer=dynamic_graph_n_layer,
                                                                         )
        graph_attention = graph_attention_layer.MultiHeadGraphAttentionLayer(hidden_size * graph_attention_n_head,
                                                                             hidden_size,
                                                                             0.2,
                                                                             leaky_alpha,
                                                                             graph_attention_n_head, )
        self.sequence_graph = graph_attention_layer.SequenceGraphFramework(
            dynamic_graph,
            graph_attention,
        )
        self.output = nn.Linear(hidden_size*graph_attention_n_head, out_vocabulary_size)

    def forward(self, encoder_sequence, initial_decoder_sequence, fixed_graph, ):
        output_begin_idx = encoder_sequence.shape[1]
        node_representation = self.input_embedding(torch.cat((encoder_sequence, initial_decoder_sequence), dim=1))
        del encoder_sequence, initial_decoder_sequence
        node_representation = self.hidden_state_transform(node_representation)
        for _ in range(self._graph_itr):
            node_representation = self.sequence_graph(node_representation, fixed_graph)
        o = self.output(node_representation[:, output_begin_idx:, :])
        return o


class PreprocessWrapper(nn.Module):
    def __init__(self,
                 m,
                 pad_idx,
                 hole_idx,
                 begin_idx):
        super().__init__()
        self.m = m
        self._pad_idx = pad_idx
        self._hole_idx = hole_idx
        self._begin_idx = begin_idx

    def forward(self, batch_data):
        return self.m(*self._preprocess(batch_data))

    def _preprocess(self, batch_data):
        from common.util import PaddedList
        intput_seq = batch_data["input"]
        encoder_sequence = PaddedList(intput_seq, self._pad_idx)
        encoder_sequence = to_cuda(autograd.Variable(torch.LongTensor(encoder_sequence)))
        max_word_length = max(len(t) for t in more_itertools.flatten(intput_seq))

        batch_size = len(intput_seq)
        max_decoder_length = batch_data['max_decoder_length'][0]
        initial_decoder_sequence = np.ones((1, max_decoder_length), dtype=np.int)
        initial_decoder_sequence *= self._hole_idx
        initial_decoder_sequence[0] = self._begin_idx
        initial_decoder_sequence = np.repeat(initial_decoder_sequence, batch_size, axis=0)
        initial_decoder_sequence = to_cuda(autograd.Variable(torch.LongTensor(initial_decoder_sequence)))\
            .unsqueeze(-1) \
            .expand(-1, -1, max_word_length)

        adj_matrix = []
        max_length = max(len(x) for x in intput_seq)
        size = max_length + max_decoder_length
        # print("size:{}".format(size))
        for i in range(batch_size):
            length = len(intput_seq[i])
            idx, idy = create_sequence_node_link(0, length)
            id1, id2 = create_sequence_node_link(max_length, max_decoder_length)
            idx.extend(id1)
            idy.extend(id2)
            adj_matrix.append(sparse.coo_matrix((np.ones((length-1+max_decoder_length-1)*2+length+max_decoder_length,
                                                         dtype=np.float),
                                                 (np.concatenate(idx),
                                                  np.concatenate(idy))),
                                                shape=(size, size), dtype=np.float).toarray())

        adj_matrix = to_cuda(torch.FloatTensor(np.stack(adj_matrix, axis=0)))
        return encoder_sequence, initial_decoder_sequence, adj_matrix


class SequenceEncoderDecoderModel(nn.Module):
    def __init__(self,
                 cfg,
                 vocab,
                 n_ctx):
        super().__init__()
        self.transformer_model = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.o = nn.Linear(cfg['n_embd'], vocab)

    def forward(self, x, output_mask):
        x = self.transformer_model(x)
        return self.o(x), output_mask


def sequence_encoder_decoder_loss():
    Loss = nn.CrossEntropyLoss()
    def loss(log_probs, target):
        output_mask = log_probs[1].unsqueeze(-1).byte()
        log_probs = log_probs[0]
        classes = log_probs.shape[-1]
        log = torch.masked_select(log_probs, output_mask).view(-1, classes)
        return Loss(log,
                    torch.cat(target))
    return loss


class SequencePreprocesser(nn.Module):
    def __init__(self,
                 m,
                 hole_idx,
                 begin_idx,
                 delimeter_idx,
                 max_length,
                 position_embedding_base,
                 ):
        super().__init__()
        self.hole_idx = hole_idx
        self.begin_idx = begin_idx
        self.delimeter_idx = delimeter_idx
        self.max_length = max_length
        self.m = m
        self._position_range = np.arange(max_length).reshape(-1, max_length) + position_embedding_base

    def _preprocess(self, x):
        def to(x):
            return to_cuda(torch.LongTensor(x))
        x = x['x']
        s = [[self.begin_idx] + t + [self.delimeter_idx] for t in x]
        batch_size = len(x)
        output_mask = [[0]*(len(t)+2)+[1]*(len(t)+1) for t in x]
        return [torch.cat([to(PaddedList(s, fill_value=self.hole_idx, shape=[batch_size, self.max_length],),).unsqueeze(-1),
                           to(np.repeat(self._position_range, batch_size, axis=0)).unsqueeze(-1)], dim=-1),
                to(PaddedList(output_mask, fill_value=0, shape=[batch_size, self.max_length]))]

    def forward(self, x):
        return self.m(*self._preprocess(x))


class SequenceEncoderDecoderModelUseEncodePad(nn.Module):
    def __init__(self,
                 cfg,
                 vocab,
                 n_ctx,
                 encoder_length):
        super().__init__()
        self.transformer_model = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.o = nn.Linear(cfg['n_embd'], vocab)
        self.encoder_length = encoder_length

    def forward(self, x,):
        x = self.transformer_model(x)
        return self.o(x[:, self.encoder_length+1:, :])


class SequencePreprocesserWithInputPad(nn.Module):
    def __init__(self,
                 m,
                 hole_idx,
                 begin_idx,
                 delimeter_idx,
                 max_length,
                 pad_idx,
                 position_embedding_base,
                 ):
        super().__init__()
        self.hole_idx = hole_idx
        self.begin_idx = begin_idx
        self.delimeter_idx = delimeter_idx
        self.max_length = max_length
        self.m = m
        self.pad_idx = pad_idx
        self._position_range = np.arange(max_length*2).reshape(-1, 2*max_length) + position_embedding_base

    def _preprocess(self, x):
        def to(x):
            return to_cuda(torch.LongTensor(x))
        x = x['x']
        s = [[self.begin_idx] + t for t in x]
        batch_size = len(x)
        content_seq = torch.cat(
                (to(PaddedList(s, fill_value=self.pad_idx, shape=[batch_size, self.max_length], ), ),
                 to(PaddedList(np.repeat(np.array([self.delimeter_idx]).reshape(1, 1), batch_size, axis=0),
                               fill_value=self.hole_idx, shape=[batch_size, self.max_length]))),
                dim=1).unsqueeze(-1)
        position_seq = to(np.repeat(self._position_range, batch_size, axis=0)).unsqueeze(-1)
        return [torch.cat(
            [content_seq,
             position_seq], dim=-1),
        ]

    def forward(self, x):
        return self.m(*self._preprocess(x))


def extract_state(x, x_state, state_idx):
    """
    :param x: [batch, seq, 2]
    :param x_state: [batch, seq, dim]
    :param state_idx: int
    :return:
    """
    dim = x_state.shape[2]
    idx = x[:, :, 0].contiguous().view(-1)
    x_state = x_state.view(-1, dim)
    return x_state[idx == state_idx, :]


class DecoderInitialState(nn.Module):
    def __init__(self,
                 cfg,
                 vocab,
                 n_ctx,
                 decoder_init_idx,
                 ):
        super().__init__()
        self.embedding_dim = cfg.n_embd
        self.decoder_initial_fill_transform = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.embedding = self.decoder_initial_fill_transform.embed
        self.decoder_init_idx = decoder_init_idx
        self.decoder_init_state_transform = FeedForwardLayer(self.embedding_dim, self.embedding_dim, 0.1, gelu)
        self.initial_state = FeedForwardLayer(self.embedding_dim*2, self.embedding_dim, 0.1, gelu)

    def forward(self, init_state_x, y_position):
        """
        :param init_state_x: [batch, max_encoder_seq+1, 2]
        :param y_position: [batch, max_decoder_seq]
        :return:
        """
        decoder_initial_state = self.decoder_initial_fill_transform(init_state_x)
        decoder_initial_state = extract_state(init_state_x, decoder_initial_state, self.decoder_init_idx)
        decoder_initial_state = self.decoder_init_state_transform(decoder_initial_state)
        # [batch, embedding]
        y_position = self.embedding(y_position)
        # [batch, seq, dim]
        decoder_initial_state = decoder_initial_state.unsqueeze(1).expand(-1, y_position.shape[1], -1)
        decoder_initial_state = torch.cat((decoder_initial_state, y_position), dim=-1)
        return self.initial_state(decoder_initial_state)


class SEDWithInitialState(nn.Module):
    def __init__(self,
                 cfg,
                 vocab,
                 n_source_ctx,
                 n_ctx,
                 decoder_init_idx,
                 ):
        super().__init__()
        self.initial_state = DecoderInitialState(cfg, vocab, n_source_ctx+1, decoder_init_idx)
        self.decoder = Transformer(cfg, n_ctx=n_ctx)
        self.position_embedding = PositionEmbedding(cfg.n_embd, vocab)
        self.o = nn.Linear(cfg.n_embd, vocab)

    def forward(self, init_state_x, x, y_position):
        y_initial_state = self.initial_state(init_state_x, y_position)
        x = self.position_embedding(x)
        y = self.position_embedding.embed(y_position) + y_initial_state
        h = torch.cat((x, y), dim=1)
        h = self.decoder(h)
        return self.o(h[:, x.shape[1]:, :])


class SEDWithInitialStatePreproceser(nn.Module):
    def __init__(self,
                 m,
                 begin_idx,
                 delimeter_idx,
                 summary_idx,
                 source_ctx,
                 pad_idx,
                 position_embedding_base,
                 ):
        super().__init__()
        self.m = m
        self.begin_idx = begin_idx
        self.delimeter_idx = delimeter_idx
        self.summary_idx = summary_idx
        self.source_ctx = source_ctx
        self.pad_idx = pad_idx
        self.source_pos = np.arange(source_ctx+1).reshape(-1, source_ctx+1) + position_embedding_base
        self.target_pos = np.arange(source_ctx-1).reshape(-1, source_ctx-1) + position_embedding_base + source_ctx + 1

    def _preprocess(self, x):
        def to(x):
            return to_cuda(torch.LongTensor(x))
        x = x['x']
        batch_size = len(x)
        init_state_x = to(PaddedList([[self.begin_idx] + t + [self.summary_idx] for t in x],
                                     fill_value=self.pad_idx,
                                     shape=(batch_size, self.source_ctx+1)))
        init_state_x = torch.cat((
            init_state_x.unsqueeze(-1),
            to(np.repeat(self.source_pos, batch_size, axis=0)).unsqueeze(-1)
        ), dim=-1)
        x = to(PaddedList([[self.begin_idx] + t for t in x], fill_value=self.pad_idx, shape=(batch_size, self.source_ctx)))
        x = torch.cat((x, to_cuda(torch.ones(batch_size, 1).long())*self.delimeter_idx), dim=1)
        x = torch.cat((
            x.unsqueeze(-1),
            to(np.repeat(self.source_pos, batch_size, axis=0)).unsqueeze(-1)
        ), dim=-1)
        y_position = to(np.repeat(self.target_pos, batch_size, axis=0))
        return init_state_x, x, y_position

    def forward(self, x):
        return self.m(*self._preprocess(x))
