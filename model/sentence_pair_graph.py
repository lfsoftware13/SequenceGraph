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

from common.character_embedding import CharacterEmbedding
from common.feed_forward_block import MultiLayerFeedForwardLayer
from common.graph_attention_layer import GGNN
from common.input_embedding import InputEmbedding, RandomInitialInputEmbedding
from common.multihead_atten import PositionwiseFeedForward, MultiHeadAttention
from common.problem_util import to_cuda
from common import util, graph_attention_layer
from common.torch_util import Attention, create_sequence_length_mask
from model.transformer_lm import DoubleHeadModel


class SequenceGraphModel(nn.Module):
    def __init__(self,
                 word_embedding: np.array,
                 character_number, character_embedding_dim=32, character_n_filters=200,
                 character_kernel_size=5, character_padding=2, n_link_type=10, hidden_state_size=32,
                 n_dynamic_link_layer=2, n_fix_graph=2, graph_itr=5, n_classes=2, tie_weight=True, summary_node=False,
                 fix_iter=2,
                 graph_propgate_method="ggnn", ):
        super().__init__()
        self.character_embedding = CharacterEmbedding(character_number, character_embedding_dim, character_n_filters,
                                                      character_kernel_size, character_padding)
        self.word_embedding = InputEmbedding(word_embedding, word_embed_dim=word_embedding.shape[1], )
        # self.base = nn.Parameter(torch.randn(1), requires_grad=True)
        if not tie_weight:
            self.dynamic_link_predictor = nn.ModuleList([nn.ModuleList(
                [graph_attention_layer.DynamicGraphAdjacentMatrix(hidden_state_size * n_link_type,
                                                                  n_dynamic_link_layer)
                 for _ in range(n_link_type-n_fix_graph)]) for _ in range(graph_itr)])
            self.multi_type_graph = nn.ModuleList([graph_attention_layer.MultiLinkTypeGGNN(n_link_type,
                                                                            hidden_state_size * n_link_type,
                                                                            hidden_state_size) for _ in range(graph_itr)])
        else:
            self.dynamic_link_predictor = nn.ModuleList(
                [graph_attention_layer.DynamicGraphAdjacentMatrix(hidden_state_size * n_link_type,
                                                                  n_dynamic_link_layer)
                 for _ in range(n_link_type-n_fix_graph)])
            self.multi_type_graph = graph_attention_layer.MultiLinkTypeGGNN(n_link_type,
                                                                            hidden_state_size * n_link_type,
                                                                            hidden_state_size)
        self.transformer = nn.Linear(hidden_state_size * n_link_type, hidden_state_size * n_link_type)
        self.tie_weight = tie_weight
        self.graph_itr = graph_itr
        self.summary_node = summary_node
        self.embedding_pad = nn.Parameter(torch.zeros(1, 1, hidden_state_size*n_link_type-word_embedding.shape[1]-character_n_filters),
                                          requires_grad=False)
        self.n_classes = n_classes
        o_input_dim = hidden_state_size * n_link_type * (graph_itr + 1)
        if summary_node:
            o_input_dim *= 4
        if n_classes == 2:
            self.o = nn.Linear(o_input_dim, 1)
        else:
            self.o = nn.Linear(hidden_state_size * n_link_type, n_classes)

    def forward(self, sentences, sentences_char, distance_matrix, ):
        distance_matrix = torch.exp(distance_matrix)
        sentences = self.word_embedding(sentences)
        sentences_char = self.character_embedding(sentences_char)
        hidden = torch.cat((
            sentences,
            sentences_char,
            self.embedding_pad.expand(sentences.shape[0], sentences.shape[1], -1)
        ), dim=-1)
        hidden_list = [hidden]
        for i in range(self.graph_itr):
            if not self.tie_weight:
                adjs = [dynamic_link_predictor(hidden) for dynamic_link_predictor in self.dynamic_link_predictor[i]]
                hidden = self.multi_type_graph[i]([distance_matrix, ] + adjs, hidden) + hidden
            else:
                adjs = [dynamic_link_predictor(hidden) for dynamic_link_predictor in self.dynamic_link_predictor]
                hidden = self.multi_type_graph([distance_matrix, ] + adjs, hidden) + hidden
            # hidden = self.transformer(hidden) + hidden
            hidden_list.append(hidden)
        if not self.summary_node:
            o, _ = torch.max(torch.cat(hidden_list, dim=-1), dim=1)
            # o, _ = torch.max(hidden, dim=1)
        else:
            s1_summary = torch.cat([h[:, -2, :] for h in hidden_list], dim=-1)
            s2_summary = torch.cat([h[:, -1, :] for h in hidden_list], dim=-1)
            o = torch.cat([s1_summary, s2_summary, torch.abs(s1_summary-s2_summary), s2_summary+s1_summary], dim=-1)

        if self.n_classes == 2:
            return self.o(o).squeeze(-1)
        else:
            return self.o(o)


class SequenceGraphModelWithGraphAttention(nn.Module):
    def __init__(self,
                 word_embedding: np.array,character_number,
                 mixed,
                 hidden_size,
                 graph_itr,
                 dynamic_graph_n_layer,
                 graph_attention_n_head,
                 leaky_alpha,
                 # use_sum_node=False,
                  character_embedding_dim=32, character_n_filters=200,
                 character_kernel_size=5, character_padding=2,
                 n_classes=2,
                 ):
        super().__init__()
        embedding_dim = word_embedding.shape[1] * (2 if mixed else 1)
        # self.use_sum_node = use_sum_node
        # if use_sum_node:
        #     self.summary_node_initial_embedding = nn.Parameter(
        #         autograd.Variable(torch.zeros(1, 1, embedding_dim)),
        #         requires_grad=True,
        #     )
        #     nn.init.xavier_normal_(self.summary_node_initial_embedding)
        self.character_embedding = CharacterEmbedding(character_number, character_embedding_dim, character_n_filters,
                                                      character_kernel_size, character_padding)
        self.word_embedding = InputEmbedding(word_embedding, word_embed_dim=word_embedding.shape[1], mixed=mixed)
        self.hidden_state_transform = nn.Linear(embedding_dim+character_n_filters, hidden_size*graph_attention_n_head)
        self.dynamic_graph = graph_attention_layer.DynamicGraphAdjacentMatrix(input_dim=
                                                                              hidden_size*graph_attention_n_head,
                                                                              n_layer=dynamic_graph_n_layer)
        self.graph_attention = graph_attention_layer.MultiHeadGraphAttentionLayer(hidden_size*graph_attention_n_head,
                                                                                  hidden_size,
                                                                                  0.2,
                                                                                  leaky_alpha,
                                                                                  graph_attention_n_head)
        if n_classes == 2:
            self.output = nn.Linear(hidden_size*graph_attention_n_head, 1)
        else:
            self.output = nn.Linear(hidden_size*graph_attention_n_head, n_classes)
        self.n_classes = n_classes
        self._graph_itr = graph_itr

    def forward(self, sentences, sentences_char, fixed_graph):
        batch_size = sentences.shape[0]
        fixed_graph = torch.exp(fixed_graph)
        sentences = self.word_embedding(sentences)
        sentences_char = self.character_embedding(sentences_char)
        node_representation = torch.cat((
            sentences,
            sentences_char,
        ), dim=-1)
        # if self.use_sum_node:
        #     node_representation = torch.cat(
        #         [q1, q2, self.summary_node_initial_embedding.expand(batch_size, -1, -1)], dim=1)
        # else:
        #     node_representation = torch.cat(
        #         [q1, q2], dim=1)
        # del q1, q2
        node_representation = self.hidden_state_transform(node_representation)
        for _ in range(self._graph_itr):
            dynamic_graph = self.dynamic_graph(node_representation)
            dynamic_graph = fixed_graph+dynamic_graph
            node_representation = node_representation + self.graph_attention(node_representation, dynamic_graph)

        # if self.use_sum_node:
        #     return self.output(node_representation[:, -1, :]).squeeze(-1)
        # else:
        o, _ = torch.max(node_representation, dim=1)
        if self.n_classes == 2:
            return self.output(o).squeeze(-1)
        else:
            return self.output(o)


class PreprocessWrapper(nn.Module):
    def __init__(self,
                 m,
                 pad_idx,
                 character_pad_idx,
                 summary_node=False,
                 ):
        super().__init__()
        self.m = m
        self._pad_idx = pad_idx
        self._character_pad_idx = character_pad_idx
        self._summary_node = summary_node

    def forward(self, batch_data):
        return self.m(*self._preprocess(batch_data))

    def _preprocess(self, batch_data):
        from common.util import PaddedList
        s1 = batch_data["s1"]
        s2 = batch_data['s2']

        batch_size = len(s1)
        size = max(len(t1)+len(t2)+1 for t1, t2 in zip(s1, s2))
        if self._summary_node:
            size += 2
        # print("size:{}".format(size))

        if not self._summary_node:
            sentences = to_cuda(torch.LongTensor(
                PaddedList([t1 + [self._pad_idx] + t2 for t1, t2 in zip(s1, s2)], fill_value=self._pad_idx,)))

            sentences_char = to_cuda(torch.LongTensor(
                PaddedList([t1 + [[self._character_pad_idx]] + t2 for t1, t2 in zip(batch_data['s1_char'], batch_data['s2_char'])],
                           fill_value=self._character_pad_idx)))
        else:
            sentences = to_cuda(torch.LongTensor(
                PaddedList([t1 + [self._pad_idx] + t2 + [self._pad_idx, self._pad_idx] for t1, t2 in zip(s1, s2)],
                           fill_value=self._pad_idx, )))

            sentences_char = to_cuda(torch.LongTensor(
                PaddedList(
                    [t1 + [[self._character_pad_idx]] + t2 + [[self._character_pad_idx], [self._character_pad_idx]] for
                     t1, t2 in
                     zip(batch_data['s1_char'], batch_data['s2_char'])],
                    fill_value=self._character_pad_idx)))

        distance_matrix = np.ones((batch_size, size, size)) * float('-inf')
        for i, (t1, t2) in enumerate(zip(s1, s2)):
            s1_matrix = util.create_distance_node_matrix(len(t1))
            s2_matrix = util.create_distance_node_matrix(len(t2))
            distance_matrix[i, :len(t1), :len(t1)] = s1_matrix
            distance_matrix[i, len(t1)+1:len(t1)+len(t2)+1, len(t1)+1:len(t1)+len(t2)+1] = s2_matrix
            if self._summary_node:
                distance_matrix[i, :len(t1), -2] = 0
                distance_matrix[i, len(t1)+1:len(t1)+len(t2)+1, -1] = 0

        distance_matrix = to_cuda(torch.FloatTensor(np.stack(distance_matrix, axis=0)))

        # sentence_same_token_link_matrix = []
        # for t1, t2 in zip(s1, s2):
        #     idx, idy, data = util.create_sentence_pair_same_node_matrix(t1, 0, t2, len(t1)+1)
        #     sentence_same_token_link_matrix.append(
        #         sparse.coo_matrix(
        #             (data, (idx, idy)),
        #             shape=(size, size), dtype=np.float
        #         ).toarray()
        #     )
        # sentence_same_token_link_matrix = to_cuda(torch.FloatTensor(np.stack(sentence_same_token_link_matrix, axis=0)))

        return sentences, sentences_char, distance_matrix,


class GGNNGraphModel(nn.Module):
    def __init__(self,
                 word_embedding: np.array,
                 max_length,
                 hidden_state_size,
                 n_classes,
                 ):
        super().__init__()
        max_length += 2
        self.word_embedding = InputEmbedding(word_embedding, word_embed_dim=word_embedding.shape[1], mixed=False,
                                             trainable=True)
        self.position_embedding = RandomInitialInputEmbedding(max_length, embedding_dim=word_embedding.shape[1])
        self._embedding_size = word_embedding.shape[1]
        self._hidden_state_size = hidden_state_size
        self._max_length = max_length
        if self._hidden_state_size > self._embedding_size:
            self.tr = nn.Linear(self._embedding_size, self._hidden_state_size)
        self.atten = MultiHeadAttention(1, self._hidden_state_size, self._hidden_state_size, self._hidden_state_size)
        # self.atten = Attention(self._hidden_state_size)
        self.ggnn_layer = GGNN(hidden_state_size)
        self.ll = MultiLayerFeedForwardLayer(4, self._hidden_state_size, self._hidden_state_size,
                                             self._hidden_state_size, 0.1)
        self.pl = PositionwiseFeedForward(self._hidden_state_size, self._hidden_state_size*4)
        if n_classes == 2:
            self.o = nn.Linear(self._hidden_state_size, 1)
        else:
            self.o = nn.Linear(self._hidden_state_size, n_classes)
        self.n_classes = n_classes

    def forward(self, s, length, position):
        s = self.word_embedding(s) + self.position_embedding(position)
        if self._hidden_state_size > self._embedding_size:
            s = self.tr(s)
        # _, link_predict, _ = self.atten(s, s, )
        _, link_predict = self.atten(s, s, s, )
        s = self.ggnn_layer(s, link_predict) + s
        # s = self.pl(s)
        s = s.view(-1, self._hidden_state_size)
        summary_node_index = torch.range(0, length.shape[0]-1, ).long().to(s.device)
        summary_node_index = summary_node_index * self._max_length + length - 1
        o = torch.index_select(s, dim=0, index=summary_node_index)
        o = self.ll(o)
        if self.n_classes == 2:
            return self.o(o).squeeze(-1)
        else:
            return self.o(o)


class TestModel(nn.Module):
    def __init__(self, cfg, clf_token, vocabulary_size, n_ctx):
        super().__init__()
        self.m = DoubleHeadModel(cfg, clf_token, vocabulary_size, n_ctx)

    def forward(self, x):
        return self.m(x)[1]


class ConcatPreprocessWrapper(nn.Module):
    def __init__(self,
                 m,
                 pad_idx,
                 delimeter_idx,
                 summary_node_idx,
                 max_length,
                 ):
        super().__init__()
        self.m = m
        self._pad_idx = pad_idx
        self._delimeter_idx = delimeter_idx
        self._summary_node_idx = summary_node_idx
        max_length += 2
        self._max_length = max_length
        self._position_range = np.arange(max_length).reshape(-1, max_length)

    def _preprocess(self, batch_data):
        from common.util import PaddedList
        s1 = batch_data["s1"]
        s2 = batch_data['s2']
        batch_size = len(s1)
        s = [t1[:-1]+[self._delimeter_idx]+t2[1:-1]+[self._summary_node_idx] for t1, t2 in zip(s1, s2)]
        # summary_node_index = []
        # for t, i in zip(s, range(batch_size)):
        #     summary_node_index.append(len(t)+i * self._max_length - 1)
        length = [len(t) for t, i in zip(s, range(batch_size))]
        for t in length:
            assert t <= self._max_length

        def to(x):
            return to_cuda(torch.LongTensor(x))

        return [torch.cat([to(PaddedList(s, fill_value=self._pad_idx, shape=[batch_size, self._max_length], )).unsqueeze(-1),
                          to(np.repeat(self._position_range, batch_size, axis=0)).unsqueeze(-1)], dim=-1)]
        # return to(PaddedList(s, fill_value=self._pad_idx, shape=[batch_size, self._max_length],)), \
        #        to(length), \
        #        to(np.repeat(self._position_range, batch_size, axis=0))

    def forward(self, batch_data):
        return self.m(*self._preprocess(batch_data))