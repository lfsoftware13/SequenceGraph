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

from common.problem_util import to_cuda
from common import util


class SequenceGraphModel(nn.Module):
    def __init__(self,
                 word_embedding: np.array):
        super().__init__()

    def forward(self, *input):
        pass


class PreprocessWrapper(nn.Module):
    def __init__(self,
                 m,
                 pad_idx,
                 ):
        super().__init__()
        self.m = m
        self._pad_idx = pad_idx

    def forward(self, batch_data):
        return self.m(*self._preprocess(batch_data))

    def _preprocess(self, batch_data):
        from common.util import PaddedList
        s1 = batch_data["s1"]
        s2 = batch_data['s2']

        batch_size = len(s1)
        size = max(len(t1)+len(t2)+1 for t1, t2 in zip(s2, s2))

        sentences = to_cuda(torch.LongTensor(
            PaddedList([t1 + [self._pad_idx] + t2 for t1, t2 in zip(s1, s2)], fill_value=self._pad_idx)))

        sentences_char = to_cuda(torch.LongTensor(
            PaddedList([t1 + [[self._pad_idx]] + t2 for t1, t2 in zip(batch_data['s1_char'], batch_data['s2_char'])],
                       fill_value=self._pad_idx)))

        distance_matrix = np.ones((batch_size, size, size)) * float('-inf')
        for i, (t1, t2) in enumerate(zip(s1, s2)):
            s1_matrix = util.create_distance_node_matrix(len(t1))
            s2_matrix = util.create_distance_node_matrix(len(t2))
            distance_matrix[i, :len(t1), :len(t1)] = s1_matrix
            distance_matrix[i, len(t1)+1:len(t1)+len(t2)+1, len(t1)+1:len(t1)+len(t2)+1] = s2_matrix

        distance_matrix = to_cuda(torch.FloatTensor(np.stack(distance_matrix, axis=0)))

        sentence_same_token_link_matrix = []
        for t1, t2 in zip(s1, s2):
            idx, idy, data = util.create_sentence_pair_same_node_matrix(t1, 0, t2, len(t1)+1)
            sentence_same_token_link_matrix.append(
                sparse.coo_matrix(
                    (data, (idx, idy)),
                    shape=(size, size), dtype=np.float
                ).toarray()
            )

        return sentences, sentences_char, distance_matrix, sentence_same_token_link_matrix
