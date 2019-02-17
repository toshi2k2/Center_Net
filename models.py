from __future__ import print_function

# from utils import RangeNormalize
import random
import math
import torch
import torch.nn as nn
# import torch.legacy.nn as lnn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
from torch.autograd import Variable
import numpy as np
# import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.distributions.normal import Normal

# normalize_01 = RangeNormalize(0,1)

class Net(nn.Module):

    def __init__(self, insize=256, output_size=128, nc=3, hidden_size=64, n_layers=2):
        super(Net, self).__init__()

        self.insize = insize
        self.hidden_size = hidden_size
        # self.output_size = output_size
        self.n_layers = n_layers

        n = math.log2(self.insize)
        self.idx = 1

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 6, 'imageSize must be at least 8'
        n = int(n)
        self.main = nn.Sequential()
        self.main2 = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv1d(nc, self.hidden_size, kernel_size=2, stride=2))
        self.main.add_module('leaky-relu', nn.LeakyReLU(0.2, inplace=True))
        self.main.add_module('dropout', nn.Dropout(0.2))
        # state size. (ndf) x 32 x 32
        for i in range(n - 6):
            self.idx += 1
            self.main.add_module('pyramid{0}-conv'.format(i + 1), nn.Conv1d(self.hidden_size * (i + 1),
                                                                            self.hidden_size * (i + 2), kernel_size=2,
                                                                            stride=2))
            self.main.add_module('pyramid{0}batchnorm'.format(i + 1), nn.BatchNorm1d(self.hidden_size * (i + 2)))
            self.main.add_module('pyramid{0}relu'.format(i + 1), nn.LeakyReLU(0.1, inplace=True))

        self.g = nn.GRU(self.hidden_size * self.idx, self.hidden_size * self.idx, self.n_layers, dropout=0.1)

        self.main2.add_module('output-linear1', nn.Linear(self.hidden_size * self.idx, 2))
        #         self.main2.add_module('output-relu', nn.LeakyReLU(0.1, inplace=True))
        #         self.main2.add_module('output-linear1', nn.Linear(self.output_size, 1))
        self.main2.add_module('output-sigmoid', nn.Sigmoid())

    def forward(self, inputs, hidden=None):
        # batch_size = inputs.size(1)
        # print(inputs.shape)
        p = self.main(inputs)
        #         print(p.shape)

        # Turn (batch_size x hidden_size x seq_len) into (seq_len x batch_size x hidden_size) for RNN
        p = p.transpose(1, 2).transpose(0, 1)
        output, hidden_out = self.g(p, hidden)
        # conv_seq_len = output.size(0)
        #         print(output.shape)

        #         output = output.view(conv_seq_len * batch_size, self.hidden_size*self.idx)  # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        #         print(output.shape)
        output = self.main2(output[-1])

        return output