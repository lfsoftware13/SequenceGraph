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

from common.util import show_process_map


class OriDataSet(Dataset):
    def __init__(self,
                 data_list,
                 transform=None):
        self._data = data_list
        if transform:
            self._data = show_process_map(transform, self._data)
            self._data = list(filter(lambda x: x is not None, self._data))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)