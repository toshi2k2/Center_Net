from __future__ import print_function

from models import Net
from train import _train
from utils import weights_init, Denormalize, RangeNormalize, to_scalar

import argparse
import random
import math
import torch
import torch.nn as nn
import pandas as pd
# import torch.legacy.nn as lnn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as vdata
from torch.utils.data import Dataset, DataLoader
# import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F
# from torch.distributions.normal import Normal

import warnings


def main():
    warnings.filterwarnings("ignore")

    #######################################################################################################################
    """Command line interface"""

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=False, help='cifar10 | mnist | fmnist '| svhn', default='mnist')
    parser.add_argument('--dataroot', required=False, help='path to dataset', default='./data/data.csv')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--len', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--saveInt', type=int, default=14, help='number of epochs between checkpoints')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--net', default='', help="path to net (to continue training)")
    parser.add_argument('--nc', default=2, help="number of channels", type=int)

    opt = parser.parse_args()
    print(opt)
    ######################################################################################################################

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if opt.cuda else "cpu")
    ######################################################################################################################
    """Dataset loading"""

    my_x = []  # a list of numpy arrays
    for _ in range(1000):
        my_x.append(np.random.rand(12, 128))
    my_y = list(np.random.randint(2, size=1000))  # another list of numpy arrays (targets)
    # print(type(my_y))

    tensor_x = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
    tensor_y = torch.Tensor(my_y)

    my_dataset = vdata.TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(my_dataset, batch_size=opt.batchSize, drop_last=True)  # create your dataloader

    test_x = []  # a list of numpy arrays
    for _ in range(100):
        test_x.append(np.random.rand(12, 128))
    test_y = list(np.random.randint(2, size=100))  # another list of numpy arrays (targets)
    # print(type(my_y))

    tensor_test_x = torch.stack([torch.Tensor(i) for i in test_x])  # transform to torch tensors
    tensor_test_y = torch.Tensor(test_y)

    tataset = vdata.TensorDataset(tensor_test_x, tensor_test_y)  # create your datset
    testloader = DataLoader(tataset, batch_size=opt.batchSize, drop_last=True)  # create your dataloader

    #######################################################################################################################

    """Hyperparameters"""

    #######################################################################################################################

    net = Net(insize=opt.len, output_size=128, nc=opt.nc, hidden_size=64, n_layers=2)
    net.apply(weights_init)

    print(net)

    if opt.net != '':
        net.load_state_dict(torch.load(opt.net))
    # print(net)

    # criterion = nn.MSELoss()  # one-hot in train
    criterion = nn.CrossEntropyLoss()  # remove one-hot in train

    _train(opt=opt, net=net, criterion=criterion, dataloader=dataloader, testloader=testloader)


if __name__ == '__main__':
    main()
