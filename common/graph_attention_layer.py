import more_itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.feed_forward_block import MultiLayerFeedForwardLayer
from common.highway import Highway
from common import torch_util


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=np.sqrt(2.0))
        self.a = nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a.weight, np.sqrt(2.0))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = self.W(input) # [batch, seq, dim]
        N = h.size()[1]
        batch_size = h.size()[0]

        m1 = torch_util.repeatRowsTensor(h, N)
        m2 = h.repeat(1, N, 1)

        a_input = torch.cat([m1, m2], dim=2)
        del m1, m2
        e = self.leakyrelu(self.a(a_input).squeeze(-1)).view(batch_size, N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = attention * adj
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        return x


class DynamicGraphAdjacentMatrix(nn.Module):
    def __init__(self, input_dim, n_layer, dropout=0.2, activator=nn.ReLU()):
        super().__init__()
        self._input_dim = input_dim
        self.structure_information_extracter = Highway(input_size=input_dim, n_layers=n_layer)
        self.link_weight = nn.Sequential(
            *(
                    [nn.Linear(input_dim*2, input_dim),
                     nn.ReLU(), nn.Dropout(dropout), ] +
                    list(more_itertools.flatten([[nn.Linear(input_dim, input_dim),
                                                  nn.ReLU(), nn.Dropout(dropout)]
                                                 for i in
                                                 range(n_layer - 2)])) +
                    [nn.Linear(input_dim, 1), activator]
            )
        )

    def forward(self, node_representation):
        node_representation = self.structure_information_extracter(node_representation)
        batch_size = node_representation.shape[0]
        max_seq_length = node_representation.shape[1]
        m1 = torch_util.repeatRowsTensor(node_representation, max_seq_length)
        m2 = node_representation.repeat(1, max_seq_length, 1)
        node_representation = torch.cat([m1, m2], dim=2)
        del m1, m2
        out = self.link_weight(node_representation)
        return out.view(batch_size, max_seq_length, max_seq_length)


class NodeRelationPrediction(nn.Module):
    def __init__(self, input_dim, n_layer, dropout=0.2, activator=nn.ReLU(), symmetry=False):
        super().__init__()
        self._input_dim = input_dim
        self.structure_information_extracter = Highway(input_size=input_dim, n_layers=n_layer)
        self.symmetry = symmetry
        self.compare_block = MultiLayerFeedForwardLayer(n_layer, 4*input_dim, 4*input_dim,
                                                        4*input_dim, dropout, activator=activator)

    def forward(self, n1, n2):
        batch_size = n1.shape[0]
        n2_length = n2.shape[1]
        n1_length = n1.shape[1]
        n1 = self.structure_information_extracter(n1)
        n2 = self.structure_information_extracter(n2)
        n1 = torch_util.repeatRowsTensor(n1, n2_length)
        n2 = n2.repeat(1, n1_length, 1)
        n = torch.cat([n1, n2, n1+n2, torch.abs(n1-n2) if self.symmetry else n1-n2], dim=2)
        return self.compare_block(n).view(batch_size, n1_length, n2_length, -1)


class SequenceGraphFramework(nn.Module):
    def __init__(self,
                 dynamic_graph_module: nn.Module,
                 graph_propagate_module: nn.Module):
        """
        output of this module is a new node representation
        :param dynamic_graph_module:
        This module
         input a node to node tensor [batch, node_number*node_number, dim],
         output is a adjacent matrix [batch, node_number, node_number]
        :param graph_propagate_module:
        This module
         input a node to node tensor [batch, node_number*node_number, dim]
         and a adjacent matrix [batch, node_number. number]
         output is the new node representation [batch, node_number, dim]
        """
        super().__init__()
        self.dynamic_graph_module = dynamic_graph_module
        self.graph_propagate_module = graph_propagate_module

    def forward(self, node_representation, fix_graph):
        # max_seq_length = node_representation.shape[1]
        # m1 = torch_util.repeatRowsTensor(node_representation, max_seq_length)
        # m2 = node_representation.repeat(1, max_seq_length, 1)
        # node_to_node_pair = torch.cat([m1, m2], dim=2)
        node_to_node_pair = node_representation
        dynamic_graph = self.dynamic_graph_module(node_to_node_pair)
        node_representation = self.graph_propagate_module(node_to_node_pair, dynamic_graph+fix_graph) + node_representation
        return node_representation


class GGNN(nn.Module):
    def __init__(self, hidden_state_size):
        super().__init__()
        self.gru_cell = nn.GRUCell(hidden_state_size, hidden_state_size)
        self.b = nn.Parameter(torch.randn(1, 1, hidden_state_size))

    def forward(self, x, adj):
        """
        :param x: shape [batch_size, seq, dim]
        :param adj: [batch_size, seq, seq]
        :return:
        """
        a = torch.bmm(adj, x) + self.b
        batch_size, seq, dim = x.shape
        o = self.gru_cell(a.view(-1, dim), x.view(-1, dim))
        return o.view(batch_size, seq, dim)


class MultiLinkTypeGGNN(nn.Module):
    def __init__(self, n_head, input_size, hidden_state, drop_out=0.2):
        super().__init__()
        ls = [nn.Linear(input_size, hidden_state) for _ in range(n_head)]
        for l in ls:
            nn.init.xavier_uniform_(l.weight, gain=np.sqrt(2.0))
        self.content_transformer = nn.ModuleList(
            [nn.Sequential(*[nn.Dropout(drop_out), l, nn.ReLU()]) for l in ls]
        )
        del ls
        self.ggnns = nn.ModuleList([GGNN(hidden_state) for _ in range(n_head)])

    def forward(self, adjs, x):
        o = torch.cat(
            [ggnn(extrater(x), adj) for adj, extrater, ggnn in zip(adjs, self.content_transformer, self.ggnns)], dim=-1)
        return o