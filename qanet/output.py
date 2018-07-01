import torch
from torch import nn
import torch.nn.functional as F


class Output(nn.Module):

    def __init__(self, input_dim=512):
        super(Output, self).__init__()

        self.d = input_dim

        self.W1 = nn.Linear(2*self.d, 1)
        self.W2 = nn.Linear(2*self.d, 1)
        self.o1 = nn.Linear(2*self.d, 3)
        # Initialize with Xavier
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.o1.weight)

    def forward(self, M0, M1, M2):

        # we use CrossEntropyLoss instead of a softmax layer here and NLLLoss
        t1 = torch.cat((M0,M1), -1)
        p1 = self.W1(t1).squeeze()
        # p2 = self.W2(torch.cat((M0,M2), -1))
        p1_softmax = F.softmax(p1, dim=-1).unsqueeze(-1)
        t1_o = torch.sum(t1*p1_softmax, dim=1)
        t1_o = self.o1(t1_o)

        return t1_o, None