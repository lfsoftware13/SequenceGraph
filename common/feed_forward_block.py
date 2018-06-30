import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from toolz.sandbox import unzip
from torch.utils.data import Dataset
from torchvision import transforms
import more_itertools
import numpy as np


class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activator=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(output_dim)
        self.activator = activator
        self.residule = (input_dim == output_dim)

    def forward(self, x):
        tmp = self.linear(x)
        shape = tmp.shape
        tmp = tmp.view(shape[0], -1, shape[-1]).permute(0, 2, 1).contiguous()
        tmp = self.dropout(self.activator(self.norm(tmp)))
        tmp = tmp.view(*shape)
        if self.residule:
            return x + tmp
        else:
            return tmp


class MultiLayerFeedForwardLayer(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_size, output_dim, dropout, activator=nn.ReLU(), last_no_activate=False):
        super().__init__()
        linear_stack = [FeedForwardLayer(input_dim, hidden_size, dropout, activator=activator)]
        linear_stack += [FeedForwardLayer(hidden_size, hidden_size, dropout, activator=activator) for _ in range(n_layer-2)]
        linear_stack += [nn.Linear(hidden_size, output_dim)] if last_no_activate else [
            FeedForwardLayer(hidden_size, output_dim, dropout, activator=activator)]
        self.feed_forward_layers = nn.Sequential(
            *linear_stack
        )

    def forward(self, x):
        return self.feed_forward_layers(x)


class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding, stride=1,
                 dropout=0.2, activator=nn.ReLU(), dimention=1, pool="max",
                 pool_kernel_size=2, pool_stride=2, pool_pad=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=output_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(output_channels)
        self.activator = activator
        self.residule = (in_channels == output_channels)
        if pool == "max":
            self.pool = nn.MaxPool2d(pool_kernel_size, stride=pool_stride, padding=pool_pad)

    def forward(self, x):
        tmp = self.dropout(self.activator(self.pool(self.norm(self.conv(x)))))
        if self.residule:
            return x + tmp
        else:
            return tmp


class MultiConvolutionLayer(nn.Module):
    def __init__(self, n_layer, in_channels, output_channels, kernel_size, padding, stride=1,
                 dropout=0.2, activator=nn.ReLU(), pool="max"):
        super().__init__()
        self.feed_forward_layers = nn.Sequential(
            *([ConvolutionLayer(in_channels, output_channels, kernel_size, padding, stride,
                                dropout, activator=activator)] +
              [ConvolutionLayer(output_channels, output_channels, kernel_size, padding, stride,
                                dropout, activator=activator) for _ in range(n_layer - 1)])
        )

    def forward(self, x):
        return self.feed_forward_layers(x)
