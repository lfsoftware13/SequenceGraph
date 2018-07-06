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
from common.input_embedding import WordCharInputEmbedding, RandomInitialInputEmbedding
from common.problem_util import to_cuda, get_gpu_index
from common.util import create_sequence_node_link, PaddedList
from model.transformer_lm import TransformerModel


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
