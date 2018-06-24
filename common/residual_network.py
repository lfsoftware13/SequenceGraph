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


class ResidualNetwork(nn.Module):
    def __init__(self, sub_module):
        super().__init__()
        self.sub_module = sub_module

    def forward(self, x):
        return x + self.sub_module(x)
