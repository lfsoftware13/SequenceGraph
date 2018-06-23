import operator
from collections import OrderedDict
from itertools import islice

import torch.nn.functional as F
import torch
import typing
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import PackedSequence

from common.util import transform_id_to_token


def save_model(model: torch.nn.Module, path):
    print('save model: {}'.format(path))
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
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort).cuda(GPU_INDEX)), length
    else:
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort)), length


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


def spilt_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).

    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer

    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    # reshape last dim
    x_shape = list(x.shape)
    # x_shape += x_shape[-1]//num_heads
    x_shape = x_shape[:-1] + [num_heads, x_shape[-1]//num_heads]
    x = x.view(x_shape)

    x = torch.transpose(x, dim0=-3, dim1=-2)
    return x


def create_sequence_length_mask(token_length, max_len, gpu_index=None):
    idxes = torch.arange(0, max_len, out=torch.Tensor(max_len)).unsqueeze(0)  # some day, you'll be able to directly do this on cuda
    if gpu_index is not None:
        idxes = idxes.cuda(gpu_index)
    # mask = autograd.Variable((trans_to_cuda(idxes) < token_length.unsqueeze(1)).float())
    mask = (idxes < token_length.unsqueeze(1).float())
    return mask


if __name__ == '__main__':
    a = []
