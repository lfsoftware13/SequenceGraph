import operator
from collections import OrderedDict
from itertools import islice

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typing
from torch.nn.modules.rnn import RNNCellBase
import numpy as np
from torch.nn.utils.rnn import PackedSequence

from common.util import transform_id_to_token


def save_model(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path, map_location={}):
    model.load_state_dict(torch.load(path, map_location=map_location))

def mask_softmax(logit, mask):
    logit = logit * mask
    logit_max, _ = torch.max(logit, dim=-1, keepdim=True)
    logit = logit - logit_max
    logit_exp = torch.exp(logit) * mask
    softmax = logit_exp/torch.sum(logit_exp, dim=-1, keepdim=True)
    return softmax


def to_sparse(x, cuda=True, gpu_index=0):
    """ converts dense tensor x to sparse format """
    print(torch.typename(x))
    x_typename = torch.typename(x).split('.')[-1]
    if cuda:
        sparse_tensortype = getattr(torch.cuda.sparse, x_typename)
    else:
        sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    if cuda:
        return sparse_tensortype(indices, values, x.size(), device=torch.device('cuda:{}'.format(gpu_index)))
    else:
        return sparse_tensortype(indices, values, x.size())


def pack_padded_sequence(padded_sequence, length, batch_firse=False,GPU_INDEX=0):
    _, idx_sort = torch.sort(length, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    length = torch.index_select(length, 0, idx_sort)
    if padded_sequence.is_cuda:
        padded_sequence = torch.index_select(padded_sequence, 0, idx_sort.cuda(GPU_INDEX))
    else:
        padded_sequence = torch.index_select(padded_sequence, 0, idx_sort)
    return torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, list(length), batch_first=batch_firse), idx_unsort


def pad_packed_sequence(packed_sequence, idx_unsort, pad_value, batch_firse=False, GPU_INDEX=0):
    padded_sequence, length = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=batch_firse,
                                                                padding_value=pad_value)
    if padded_sequence.is_cuda:
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort).cuda(GPU_INDEX)), \
               torch.index_select(length.cuda(GPU_INDEX), 0, torch.autograd.Variable(idx_unsort).cuda(GPU_INDEX))
    else:
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort)), \
               torch.index_select(length, 0, torch.autograd.Variable(idx_unsort))


def pack_sequence(sequences, GPU_INDEX=0):
    length = torch.Tensor([len(seq) for seq in sequences])
    _, idx_sort = torch.sort(length, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)
    packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
    return packed_sequences, idx_unsort


def create_ori_index_to_packed_index_dict(batch_sizes):
    begin_index = 0
    end_index = 0
    res = {}
    for i in range(len(batch_sizes)):
        end_index += batch_sizes[i]
        for j in range(end_index-begin_index):
            res[(j, i)] = begin_index + j
        begin_index += batch_sizes[i]
    return res


def create_stable_log_fn(epsilon):
    def stable_log(softmax_value):
        softmax_value = torch.clamp(softmax_value, epsilon, 1.0-epsilon)
        return torch.log(softmax_value)
    return stable_log


def padded_tensor_one_dim_to_length(one_tensor, dim, padded_length, is_cuda=False, gpu_index=0, fill_value=0):
    before_encoder_shape = list(one_tensor.shape)
    before_encoder_shape[dim] = padded_length - before_encoder_shape[dim]
    expend_tensor = (torch.ones(before_encoder_shape) * fill_value)
    if is_cuda:
        expend_tensor = expend_tensor.cuda(gpu_index)
    padded_outputs = torch.cat((one_tensor, expend_tensor), dim=dim)
    return padded_outputs


class MultiRNNCell(RNNCellBase):
    def __init__(self, cell_list: typing.List[RNNCellBase]):
        super().__init__()
        for idx, module in enumerate(cell_list):
            self.add_module(str(idx), module)

    def reset_parameters(self):
        for cell in self._modules.values():
            cell.reset_parameters()

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MultiRNNCell(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, h_i, h_s):
        res_h = []
        for h, cell in zip(h_s, self._modules.values()):
            h = cell(h_i, h)
            res_h.append(h)
            if isinstance(cell, torch.nn.LSTMCell):
                h_i = h[0]
            else:
                h_i = h
        return h_i, res_h


def calculate_accuracy_of_code_completion(log_probs, target, ignore_token=None, topk_range=(1, 15), gpu_index=None):
    """
    compare the log probility of all possible token with target token. calculate the accuracy of the code.
    ensure dim[1] of log_probs(seq len) is the same as dim[1] of target.
    :param log_probs:
    :param target:
    :param ignore_token:
    :param save_name:
    :param topk_range: (min_k, max_k)
    :return:
    """
    # log_probs_size = [batch_size, seq_len, vocab]
    if isinstance(target, list):
        target = torch.LongTensor(target)
        if gpu_index is not None:
            target = target.cuda(gpu_index)
    if isinstance(log_probs, PackedSequence):
        log_probs = log_probs.data
    if isinstance(target, PackedSequence):
        target = target.data

    batch_size = log_probs.shape[0]
    vocab_size = log_probs.shape[-1]

    log_probs = log_probs.view(-1, vocab_size)
    target = target.view(-1)

    if log_probs.shape[0] != target.shape[0]:
        print('different shape between log_probs and target. log_probs: {}, target: {}'.format(log_probs.shape, target.shape))
        raise Exception('different shape between log_probs and target. log_probs: {}, target: {}'.format(log_probs.shape, target.shape))

    # if len(log_probs.shape) == 2:
    #     log_probs = log_probs.unsqueeze(dim=1)

    max_topk = max(*topk_range)
    min_topk = min(*topk_range)
    if min_topk < 1:
        min_topk = 1
    if max_topk < 1:
        max_topk = 1

    # top_k_ids_size = [batch_size, seq_len, max_topk]
    top_k_ids = torch.topk(log_probs, dim=1, k=max_topk)[1]

    # resize target to the same shape of top k ids
    target = torch.unsqueeze(target, dim=1)
    repeat_shape = [1] * len(target.shape)
    repeat_shape[-1] = max_topk
    repeat_target = target.repeat(*repeat_shape)
    equal_list = torch.eq(top_k_ids, repeat_target)

    if ignore_token is not None:
        mask = torch.ne(target, ignore_token)
        zero_tensor = torch.zeros(equal_list.shape).byte()
        if gpu_index is not None:
            zero_tensor = zero_tensor.cuda(gpu_index)
        equal_list = torch.where(mask, equal_list, zero_tensor)

    result = {}
    for k in range(min_topk, max_topk+1):
        result[k] = equal_list[:, min_topk-1:k].sum().item()
    return result


# def get_predict_and_target_tokens(log_probs, target, id_to_word_fn, k=1, offset=0):
#     _, top_k_ids = torch.topk(log_probs, dim=2, k=k)
#     top_k_ids = top_k_ids.tolist()
#     batch_predict = []
#     batch_target = []
#     for i, one in enumerate(top_k_ids):
#         # one shape = [seq_len, k]
#         predict_tokens = [transform_id_to_token(one_position, id_to_word_fn, offset=offset) for one_position in one]
#         out_token = transform_id_to_token(target[i], id_to_word_fn, offset=offset)
#         batch_predict += [predict_tokens]
#         batch_target += [out_token]
#     return batch_predict, batch_target


def get_predict_and_target_tokens(log_probs, target, id_to_word_fn, k=1, offset=0):
    dim_len = len(log_probs.shape)
    softmaxed_probs = F.softmax(log_probs, dim=dim_len - 1)
    top_k_probs, top_k_ids = torch.topk(softmaxed_probs, dim=2, k=k)
    top_k_ids = top_k_ids.tolist()
    batch_predict = []
    batch_target = []
    for i, one in enumerate(top_k_ids):
        # one shape = [seq_len, k]
        predict_tokens = [transform_id_to_token(one_position, id_to_word_fn, offset=offset) for one_position in one]
        out_token = transform_id_to_token(target[i], id_to_word_fn, offset=offset)
        batch_predict += [predict_tokens]
        batch_target += [out_token]
    return batch_predict, batch_target, top_k_probs.tolist()


def create_sequence_length_mask(token_length, max_len, gpu_index=None):
    idxes = torch.arange(0, max_len, out=torch.Tensor(max_len)).unsqueeze(0)  # some day, you'll be able to directly do this on cuda
    if gpu_index is not None:
        idxes = idxes.cuda(gpu_index)
    # mask = autograd.Variable((trans_to_cuda(idxes) < token_length.unsqueeze(1)).float())
    mask = (idxes < token_length.unsqueeze(1).float())
    return mask


class SoftMarginLossWithLogit(nn.SoftMarginLoss):

    def forward(self, input, target):
        return super().forward(input, (target-0.5)*2)

    def __init__(self, size_average=True):
        super().__init__(size_average)


class Attention(nn.Module):

    def __init__(self, hidden_size,):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, output, context, mask=None):
        # print("output contains nan:{}".format(np.any(np.isnan(output.data.cpu().numpy()))))
        # print("context contains nan:{}".format(np.any(np.isnan(context.data.cpu().numpy()))))
        # print("mask contains nan:{}".format(np.any(np.isnan(mask.data.cpu().numpy()))))
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        # print("attn contains nan:{}".format(np.any(np.isnan(attn.data.cpu().numpy()))))
        if mask is not None:
            attn.data.masked_fill_(~mask.unsqueeze(1), -float('inf'))
        # print("attn contains nan:{}".format(np.any(np.isnan(attn.data.cpu().numpy()))))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # print("attn contains nan:{}".format(np.any(np.isnan(attn.data.cpu().numpy()))))

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # print("mix contains nan:{}".format(np.any(np.isnan(mix.data.cpu().numpy()))))

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output, attn, mix


class RNNUtility(nn.Module):
    def __init__(self,
                 cell_name,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0,
                 ):
        super().__init__()
        self._cell_name = cell_name
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        if cell_name == 'lstm':
            self._rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            self._initial_state = nn.ParameterList([
                nn.Parameter(torch.randn((num_layers * num_directions, 1, hidden_size)),
                             requires_grad=True),
                nn.Parameter(torch.randn((num_layers * num_directions, 1, hidden_size)),
                             requires_grad=True)])
        elif cell_name == 'gru':
            self._rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            self._initial_state = nn.Parameter(
                torch.randn((num_layers * num_directions, 1, hidden_size)),
                requires_grad=True)
        else:
            raise ValueError("The cell name should be lstm or gru")

    def forward(self,
                input_seq,
                length=None,
                embedding=None,
                gpu_index=0,
                batch_first=True,
                swap_to_original_order=True,
                pad_value=-1):
        # print("input seq contains nan:{}".format(np.any(np.isnan(input_seq.data.cpu().numpy()))))
        if isinstance(input_seq, torch.nn.utils.rnn.PackedSequence):
            if swap_to_original_order:
                raise ValueError("The PackedSequence input cam not swap")
        elif isinstance(input_seq, list):
            packed_seq, idx_unsort = pack_sequence(input_seq, gpu_index)
        else:
            if length is None:
                raise ValueError("The input sequence is pytorch tensor, but length is None")
            packed_seq, idx_unsort = pack_padded_sequence(input_seq, length, batch_firse=batch_first,
                                                          GPU_INDEX=gpu_index)
        if embedding is not None:
            packed_seq = torch.nn.utils.rnn.PackedSequence(embedding(packed_seq.data).cuda(gpu_index),
                                                           packed_seq.batch_sizes)

        # print("input seq contains nan:{}".format(np.any(np.isnan(input_seq.data.data.cpu().numpy()))))

        batch_size = packed_seq.batch_sizes[0]
        if self._cell_name == 'lstm':
            initial_state = [t.expand(-1, batch_size, -1).contiguous() for t in self._initial_state]
        else:
            initial_state = self._initial_state.expand(-1, batch_size, -1).contiguous()

        o, hidden_state = self._rnn(packed_seq, initial_state)
        # print("output contains nan:{}".format(np.any(np.isnan(o.data.data.cpu().numpy()))))
        # print("hidden_state[0] contains nan:{}".format(np.any(np.isnan(hidden_state[0].data.cpu().numpy()))))

        if swap_to_original_order:
            o, length = pad_packed_sequence(o, idx_unsort=idx_unsort, pad_value=pad_value, batch_firse=batch_first,
                                            GPU_INDEX=gpu_index)
            # print("output contains nan:{}".format(np.any(np.isnan(o.data.cpu().numpy()))))
            if self._cell_name == 'lstm':
                hidden_state = [torch.index_select(t.transpose(0, 1), 0, torch.autograd.Variable(idx_unsort).cuda(gpu_index)) for t in
                                hidden_state]
                # print("hidden_state[0] contains nan:{}".format(np.any(np.isnan(hidden_state[0].data.cpu().numpy()))))
            elif self._cell_name == 'gru':
                hidden_state = torch.index_select(hidden_state.transpose(0, 1), 0, torch.autograd.Variable(idx_unsort).cuda(gpu_index))
            return o, hidden_state, length
        else:
            return o, hidden_state, idx_unsort


class ScaledDotProductionAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask):
        d_k = k.size()[-1]
        attn = torch.bmm(q, k.transpose(1, 2))/torch.sqrt(d_k)
        if mask is not None:
            attn.data.masked_fill_(~mask.unsqueeze(1), -float('inf'))
        attn = F.softmax(attn, dim=-1)
        mix = torch.bmm(attn, v)
        return mix


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 head_num,
                 model_dimension,
                 d_k,
                 d_v):
        super().__init__()
        self._q_linear_list = nn.ModuleList([
            nn.Linear(model_dimension, d_k, bias=False) for _ in range(head_num)
        ])
        self._k_linear_list = nn.ModuleList([
            nn.Linear(model_dimension, d_k, bias=False) for _ in range(head_num)
        ])
        self._v_linear_list = nn.ModuleList([
            nn.Linear(model_dimension, d_v, bias=False) for _ in range(head_num)
        ])
        self._scaled_dot_attention_list = nn.ModuleList({ScaledDotProductionAttention() for _ in range(head_num)})
        self._re_project_map = nn.Linear(head_num * d_v, model_dimension, bias=False)

    def forward(self, q, k, v, mask):
        o = [atten(q_map(q), k_map(k), v_map(v), mask) for q_map, k_map, v_map, atten in
             zip(self._q_linear_list, self._k_linear_list, self._v_linear_list, self._scaled_dot_attention_list)]
        return self._re_project_map(torch.cat(o, dim=-1))


def repeatRowsTensor(X, rep):
    """
    :param X: a tensor [batch,  seq, dim]
    :param rep: the repeat number
    :type rep: int
    :return: a tensor with size [batch, seq*rep, dim]. It first repeats the first row #rep times, then the second
    and so on.
    """
    (depth, _, col) = X.shape
    # Open dim after batch ("depth")
    X = torch.unsqueeze(X, 1)
    # Repeat the matrix in the dim opened ("depth")
    X = X.repeat(1, rep, 1, 1)
    # Permute depth and lines to get the repeat over lines
    X = X.permute(0, 2, 1, 3)
    # Return to input (#batch x #lines*#repeat x #cols)
    X = X.contiguous().view(depth, -1, col)

    return X


class SequenceCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, C, Q):
        """
        :param C: a tensor [batch_size, seq, dim]
        :param Q: a tensor [batch_size, seq, dim]
        :return: a tensor [batch, seq_C, seq_Q]
        """
        self.batch_size = C.shape[0]
        self.n = C.shape[1]
        self.m = Q.shape[1]
        S = self.similarity(C, Q)
        return S

    def similarity(self, C, Q):
        # Create QSim (#batch x n*m x d) where each of the m original rows are repeated n times
        QSim = repeatRowsTensor(Q, self.n)
        # Create CSim (#batch x n*m x d) where C is reapted m times
        CSim = C.repeat(1, self.m, 1)
        assert QSim.shape == CSim.shape
        Sim_col = F.cosine_similarity(QSim, CSim, dim=2)
        # Put it back in right dim
        Sim = Sim_col.view(self.batch_size, self.m, self.n).permute(0, 2, 1)

        return Sim
