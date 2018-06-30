import math

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

from common.capsules import PrimaryCapsules, RoutingCapsules, squash
from common.character_embedding import CharacterEmbedding
from common.depthwise_separable_conv import DepthwiseSeparableConv1d
from common.feed_forward_block import MultiConvolutionLayer, MultiLayerFeedForwardLayer, ConvolutionLayer
from common.graph_attention_layer import NodeRelationPrediction
from common.input_embedding import InputEmbedding
from common.problem_util import to_cuda
from common.self_attention import SelfAttention


class ConvolutionBlock(nn.Module):
    def __init__(self, n_filter=128, kernel_size=7, padding=3, n_heads=8, type="normal", dropout=0.1,
                 self_attention=True):
        super().__init__()
        if type == 'normal':
            self.conv = nn.Conv1d(in_channels=n_filter, out_channels=n_filter, kernel_size=kernel_size, padding=padding)
        elif type == 'depthwise_separable':
            self.conv = DepthwiseSeparableConv1d(n_filters=n_filter, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(num_features=n_filter)
        self.atten_norm = nn.BatchNorm1d(num_features=n_filter)
        self.atten = SelfAttention(n_heads=n_heads, n_filters=n_filter)
        self.dropout = nn.Dropout(dropout)
        self.activator = nn.ReLU()
        self._self_attention = self_attention

    def forward(self, x):
        x = self.dropout(self.activator(self.norm(self.conv(x)))) + x
        if self._self_attention:
            tmp = self.atten_norm(x).permute(0, 2, 1)
            tmp = self.atten(tmp).permute(0, 2, 1)
            x = x + tmp
        return x


class GraphClusterModel(nn.Module):
    def __init__(self, word_embedding: np.array,
                 character_number, character_embedding_dim=32, character_n_filters=200,
                 character_kernel_size=5, character_padding=2, hidden_size=128, conv_type='normal',
                 resize_kernel_size=7, resize_pad_size=3,
                 n_encoder_conv_layer=2, encoder_kernel_size=7, encoder_padding=3, n_self_attention_heads=8,
                 route_number=3, n_capsules=5, capsules_dim=256, n_compare_layer=2, n_layer_output_conv=2,
                 n_layer_output_feedforward=2, hidden_size_output_feedforward=128, n_classes=2, dropout=0.2):
        super().__init__()
        self.character_embedding = CharacterEmbedding(character_number, character_embedding_dim, character_n_filters,
                                                      character_kernel_size, character_padding)
        self.word_embedding = InputEmbedding(word_embedding, word_embed_dim=word_embedding.shape[1], )
        self.resize_conv = nn.Conv1d(in_channels=word_embedding.shape[1]+character_n_filters,
                                     out_channels=hidden_size, kernel_size=resize_kernel_size, padding=resize_pad_size)
        self.conv_list = nn.ModuleList([ConvolutionBlock(n_filter=hidden_size,
                                                         kernel_size=encoder_kernel_size,
                                                         padding=encoder_padding,
                                                         type=conv_type,
                                                         n_heads=n_self_attention_heads,
                                                         dropout=dropout)
                                        for _ in range(n_encoder_conv_layer)])
        self.capsules_routing = RoutingCapsules(in_dim=hidden_size, num_caps=n_capsules, dim_caps=capsules_dim,
                                                num_routing=route_number, )
        self.compare_block = NodeRelationPrediction(capsules_dim, n_compare_layer, symmetry=True)
        self.output_conv = nn.Sequential(
            *[ConvolutionLayer(int(capsules_dim * math.pow(2, i)), int(capsules_dim * math.pow(2, i + 1)),
                               kernel_size=encoder_kernel_size, padding=encoder_padding,
                               dropout=dropout) for i in range(n_layer_output_conv)]
        )
        self.output_fc = MultiLayerFeedForwardLayer(
            n_layer=n_layer_output_feedforward,
            input_dim=capsules_dim*n_capsules*n_capsules/math.pow(2, n_layer_output_conv),
            hidden_size=hidden_size_output_feedforward,
            output_dim=1 if n_classes == 2 else n_classes,
            dropout=dropout, last_no_activate=True
        )
        self.n_classes = n_classes

    def forward(self, q1, q1_char, q2, q2_char):
        q1, q2 = [self.word_embedding(t) for t in [q1, q2]]
        q1_char, q2_char = [self.character_embedding(t) for t in [q1_char, q2_char]]
        q1 = torch.cat((q1, q1_char), dim=-1).permute(0, 2, 1) # [batch, dim, length]
        q2 = torch.cat((q2, q2_char), dim=-1).permute(0, 2, 1)
        del q1_char, q2_char
        q1, q2 = [self.resize_conv(t) for t in (q1, q2)]
        for conv in self.conv_list:
            q1 = conv(q1)
            q2 = conv(q2)
        q1 = q1.permute(0, 2, 1)
        q2 = q2.permute(0, 2, 1)
        q1 = self.capsules_routing(squash(q1))
        q2 = self.capsules_routing(squash(q2))
        compare_matrix = self.compare_block(q1, q2).permute(0, 3, 1, 2)
        del q1, q2
        compare_matrix = self.output_conv(compare_matrix)
        batch_size = compare_matrix.shape[0]
        o = compare_matrix.view(batch_size, -1)
        del compare_matrix
        o = self.output_fc(o)
        if self.n_classes == 2:
            o = o.squeeze(-1)
        return o
