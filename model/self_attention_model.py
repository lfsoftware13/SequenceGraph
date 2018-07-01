import math
import os
import numpy as np

import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

import config
from common import torch_util
from common.character_embedding import CharacterEmbedding
from common.feed_forward_block import MultiLayerFeedForwardLayer
from common.input_embedding import InputEmbedding, InputPairEmbedding
from common.problem_util import to_cuda
from common.torch_util import spilt_heads, create_sequence_length_mask, Attention
from common.util import data_loader, show_process_map, PaddedList


def add_position_encode(x, position_start_list=None, min_timescale=1.0, max_timescale=1.0e4):
    """

    :param x: has more than 3 dims
    :param position_start_list: len(position_start_list) == len(x.shape) - 2. default: [0] * num_dims.
            create position from start to start+length-1 for each dim.
    :param min_timescale:
    :param max_timescale:
    :return:
    """
    x_shape = list(x.shape)
    num_dims = len(x_shape) - 2
    channels = x_shape[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescales_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(torch.range(0, num_timescales-1)) * -log_timescales_increment
    # add moved position start index
    if position_start_list is None:
        position_start_list = [0] * num_dims
    for dim in range(num_dims):
        length = x_shape[dim + 1]
        # position = transform_to_cuda(torch.range(0, length-1))
        # create position from start to start+length-1 for each dim
        position = torch.range(position_start_list[dim], position_start_list[dim] + length - 1)
        scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = F.pad(signal, (prepad, postpad, 0, 0))
        for _ in range(dim + 1):
            signal = torch.unsqueeze(signal, dim=0)
        for _ in range(num_dims - dim -1):
            signal = torch.unsqueeze(signal, dim=-2)
        x += signal
    return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, value_hidden_size, output_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.output_dim = output_dim
        self.value_hidden_size = value_hidden_size
        self.transform_output = nn.Linear(value_hidden_size, output_dim)

    def forward(self, query, key, value, value_mask=None):
        """
        query = [batch, -1, query_seq, query_hidden_size]
        key = [batch, -1, value_seq, query_hidden_size]
        value = [batch, -1, value_seq, value_hidden_size]
        the len(shape) of query, key, value must be the same
        Create attention value of query_sequence according to value_sequence(memory).
        :param query:
        :param key:
        :param value:
        :param value_mask: [batch, max_len]. consist of 0 and 1.
        :return: attention_value = [batch, -1, query_seq, output_hidden_size]
        """
        query_shape = list(query.shape)
        key_shape = list(key.shape)
        value_shape = list(value.shape)
        scaled_value = math.sqrt(key_shape[-1])

        # [batch, -1, query_hidden_size, value_seq] <- [batch, -1, value_seq, query_hidden_size]
        key = torch.transpose(key, dim0=-2, dim1=-1)

        # [batch, -1, query_seq, value_seq] = [batch, -1, query_seq, query_hidden_size] * [batch, -1, query_hidden_size, value_seq]
        query_3d = query.contiguous().view(-1, query_shape[-2], query_shape[-1])
        key_3d = key.contiguous().view(-1, key_shape[-1], key_shape[-2])
        qk_dotproduct = torch.bmm(query_3d, key_3d).view(*query_shape[:-2], query_shape[-2], key_shape[-2])
        scaled_qk_dotproduct = qk_dotproduct/scaled_value

        # mask the padded token value
        if value_mask is not None:
            dim_len = len(list(scaled_qk_dotproduct.shape))
            scaled_qk_dotproduct.data.masked_fill_(~value_mask.view(value_shape[0], *[1 for i in range(dim_len-2)], value_shape[-2]), -float('inf'))

        weight_distribute = F.softmax(scaled_qk_dotproduct, dim=-1)
        weight_shape = list(weight_distribute.shape)
        attention_value = torch.bmm(weight_distribute.view(-1, *weight_shape[-2:]), value.contiguous().view(-1, *value_shape[-2:]))
        attention_value = attention_value.view(*weight_shape[:-2], *list(attention_value.shape)[-2:])
        transformed_output = self.transform_output(attention_value)
        return transformed_output


class MaskedMultiHeaderAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, attention_type='scaled_dot_product'):
        super(MaskedMultiHeaderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_hidden_size = int(hidden_size/num_heads)

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        if attention_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention(self.attention_hidden_size, self.attention_hidden_size)
        else:
            raise Exception('no such attention_type: {}'.format(attention_type))

        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, memory, memory_mask=None):
        """
        query = [batch, sequence, hidden_size]
        memory = [batch, sequence, hidden_size]

        :param query:
        :param key:
        :param value:
        :return:
        """
        query = self.query_linear(inputs)
        key = self.key_linear(memory)
        value = self.value_linear(memory)

        query_shape = list(query.shape)

        split_query = spilt_heads(query, self.num_heads)
        split_key = spilt_heads(key, self.num_heads)
        split_value = spilt_heads(value, self.num_heads)

        atte_value = self.attention.forward(split_query, split_key, split_value, value_mask=memory_mask)
        atte_value = torch.transpose(atte_value, dim0=-3, dim1=-2).contiguous().view(query_shape[:-1] + [-1])

        output_value = self.output_linear(atte_value)
        return output_value


class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, hidden_layer_count=1):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param hidden_layer_count: must >= 1
        """
        super(PositionWiseFeedForwardNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        if hidden_layer_count <= 0:
            raise Exception('at least one hidden layer')
        self.ff = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        for i in range(hidden_layer_count-1):
            self.ff.add_module('hidden_' + str(i), nn.Linear(hidden_size, hidden_size))
            self.ff.add_module('relu_' + str(i), nn.ReLU())

        self.ff.add_module('output', nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.ff(x)


class SelfAttentionEncoder(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(SelfAttentionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)

        self.self_attention = MaskedMultiHeaderAttention(hidden_size, num_heads)
        self.ff = PositionWiseFeedForwardNet(hidden_size, hidden_size, hidden_size, hidden_layer_count=1)
        if normalize_type == 'layer':
            self.self_attention_normalize = nn.LayerNorm(normalized_shape=hidden_size)
            self.ff_normalize = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, input, input_mask):
        atte_value = self.dropout(self.self_attention.forward(input, input, memory_mask=input_mask))
        if self.normalize_type is not None:
            atte_value = self.self_attention_normalize(atte_value) + atte_value

        ff_value = self.dropout(self.ff.forward(atte_value))
        if self.normalize_type is not None:
            ff_value = self.ff_normalize(ff_value) + ff_value
        return ff_value


class SelfAttentionDecoder(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, num_heads=1, normalize_type=None):
        super(SelfAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.normalize_type = normalize_type
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)

        self.input_self_attention = MaskedMultiHeaderAttention(hidden_size, num_heads)
        self.attention = MaskedMultiHeaderAttention(hidden_size, num_heads)
        self.ff = PositionWiseFeedForwardNet(hidden_size, hidden_size, hidden_size, hidden_layer_count=1)

        if normalize_type is not None:
            self.self_attention_normalize = nn.LayerNorm(normalized_shape=hidden_size)
            self.attention_normalize = nn.LayerNorm(normalized_shape=hidden_size)
            self.ff_normalize = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, input, input_mask, encoder_output, encoder_mask):
        self_atte_value = self.dropout(self.input_self_attention(input, input, input_mask))
        if self.normalize_type is not None:
            self_atte_value = self.self_attention_normalize(self_atte_value)

        atte_value = self.dropout(self.attention(self_atte_value, encoder_output, encoder_mask))
        if self.normalize_type is not None:
            atte_value = self.attention_normalize(atte_value)

        ff_value = self.dropout(self.ff(atte_value))
        if self.normalize_type is not None:
            ff_value = self.ff_normalize(ff_value)

        return ff_value


class SelfAttentionPairModel(nn.Module):
    def __init__(self,
                 word_embedding: np.array,
                 character_number, character_embedding_dim=32, character_n_filters=200,
                 character_kernel_size=5, character_padding=2, self_attention_layer=4, n_classes=2):
        super().__init__()
        self.character_embedding = CharacterEmbedding(character_number, character_embedding_dim, character_n_filters,
                                                      character_kernel_size, character_padding)
        self.word_embedding = InputPairEmbedding(word_embedding, word_embed_dim=word_embedding.shape[1], )
        # self.self_attentions = nn.ModuleList([SelfAttentionEncoder(hidden_size=character_n_filters+word_embedding.shape[1], num_heads=2,
        #                                                            normalize_type='layer') for _ in range(self_attention_layer)])
        self.self_attentions = Attention(hidden_size=character_n_filters+word_embedding.shape[1])
        self.ll = MultiLayerFeedForwardLayer(4, (character_n_filters+word_embedding.shape[1])*4, 200, 200, 0.1)
        if n_classes == 2:
            self.o = nn.Linear(200, 1)
        else:
            self.o = nn.Linear(200, n_classes)
        self.n_classes = n_classes

    def forward(self, q1, q1_char, q2, q2_char):
        q1, q2 = self.word_embedding(q1, q2)
        q1_char = self.character_embedding(q1_char)
        q2_char = self.character_embedding(q2_char)
        q1 = torch.cat((q1, q1_char), dim=-1)
        q2 = torch.cat((q2, q2_char), dim=-1)
        del q1_char, q2_char

        # for self_atten in self.self_attentions:
        #     q1 = self_atten(q1, None) + q1
        #     q2 = self_atten(q2, None) + q2
        q1, _, _ = self.self_attentions(q1, q1, )
        q2, _, _ = self.self_attentions(q2, q2)

        q1_o, _ = torch.max(q1, dim=1)
        q2_o, _ = torch.max(q2, dim=1)
        o = self.ll(torch.cat((q1_o, q2_o, q1_o+q2_o, torch.abs(q1_o-q2_o)), dim=-1))
        if self.n_classes == 2:
            return self.o(o).squeeze(-1)
        else:
            return self.o(o)


class PreprocessWrapper(nn.Module):
    def __init__(self,
                 m,
                 pad_idx,
                 character_pad_idx,
                 ):
        super().__init__()
        self.m = m
        self._pad_idx = pad_idx
        self._character_pad_idx = character_pad_idx

    def forward(self, batch_data):
        return self.m(*self._preprocess(batch_data))

    def _preprocess(self, batch_data):
        from common.util import PaddedList
        s1 = batch_data["s1"]
        s2 = batch_data['s2']

        return to_cuda(torch.LongTensor(PaddedList(s1, fill_value=self._pad_idx))), \
               to_cuda(torch.LongTensor(PaddedList(batch_data['s1_char'], fill_value=self._character_pad_idx))), \
               to_cuda(torch.LongTensor(PaddedList(s2, fill_value=self._pad_idx))), \
               to_cuda(torch.LongTensor(PaddedList(batch_data['s2_char'],
                                  fill_value=self._character_pad_idx)))
