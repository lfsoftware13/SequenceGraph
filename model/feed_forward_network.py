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
import math

from common.feed_forward_block import MultiLayerFeedForwardLayer, ConvolutionLayer
from common.graph_attention_layer import NodeRelationPrediction
from common.input_embedding import InputEmbedding


class FFN(nn.Module):
    def __init__(self,
                 word_embedding: np.array,
                 n_classes,
                 ):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=word_embedding.shape[0],
                                      embedding_dim=word_embedding.shape[1])
        # Cast to float because the character embeding will be returned as a float, and we need to concatenate the two
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(word_embedding).float())
        self.word_embedding.weight.requires_grad = False
        embedding_dim = word_embedding.shape[1]
        self.l0 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )

        self.ll = MultiLayerFeedForwardLayer(4, embedding_dim*2, 200, 200, 0.1)
        if n_classes == 2:
            self.o = nn.Linear(200, 1)
        else:
            self.o = nn.Linear(200, n_classes)
        self.n_classes = n_classes

    def forward(self, q1, q1_char, q2, q2_char):
        q1 = self.word_embedding(q1)
        q1 = self.l0(q1)
        q1, _ = torch.max(q1, dim=1)
        q2 = self.word_embedding(q2)
        q2 = self.l0(q2)
        q2, _ = torch.max(q2, dim=1)
        m = torch.cat((q1, q2), dim=-1)
        if self.n_classes == 2:
            return self.o(self.ll(m)).squeeze(-1)
        else:
            return self.o(self.ll(m))


class FFNWithCrossCompare(nn.Module):
    def __init__(self,
                 word_embedding: np.array,
                 conv_layer=1,
                 hidden_size=300,
                 n_classes=3,
                 ):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=word_embedding.shape[0],
                                           embedding_dim=word_embedding.shape[1])
        # Cast to float because the character embeding will be returned as a float, and we need to concatenate the two
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(word_embedding).float())
        self.word_embedding.weight.requires_grad = False
        embedding_dim = word_embedding.shape[1]
        self.l0 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU()
        )
        self.compare = NodeRelationPrediction(hidden_size, 1, )
        self.output_conv = nn.Sequential(
            *[ConvolutionLayer(hidden_size, hidden_size,
                               kernel_size=7, padding=3,
                               dropout=0.1, pool_kernel_size=5, pool_stride=1, pool_pad=2) for i in range(conv_layer)]
        )
        self.ll = MultiLayerFeedForwardLayer(4, hidden_size, 200, 200, 0.1)
        if n_classes == 2:
            self.o = nn.Linear(200, 1)
        else:
            self.o = nn.Linear(200, n_classes)
        self.n_classes = n_classes

    def forward(self, q1, q1_char, q2, q2_char):
        q1 = self.word_embedding(q1)
        q1 = self.l0(q1)
        q2 = self.word_embedding(q2)
        q2 = self.l0(q2)
        m = self.compare(q1, q2).permute(0, 3, 1, 2)
        m = self.output_conv(m).permute(0, 2, 3, 1)
        m = m.view(m.shape[0], -1, m.shape[-1])
        m, _ = torch.max(m, dim=1)
        m = self.ll(m)
        if self.n_classes == 2:
            return self.o(m).squeeze(-1)
        else:
            return self.o(m)

