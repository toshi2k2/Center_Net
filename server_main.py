from __future__ import print_function

from models import Net
from server_predict import _test
from utils import weights_init

import argparse
import random
import math
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as vdata
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import pickle
import socket

import torch.nn.functional as F
# from torch.distributions.normal import Normal

import warnings


def main():
    warnings.filterwarnings("ignore")

    #######################################################################################################################
    """Command line interface"""

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=False, help='cifar10 | mnist | fmnist '| svhn', default='mnist')
    # parser.add_argument('--dataroot', required=False, help='path to dataset', default='./data/data.csv')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    # parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--len', type=int, default=256, help='the height / width of the input to network')
    # parser.add_argument('--saveInt', type=int, default=14, help='number of epochs between checkpoints')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    # parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--net', help="path to net (to continue training)", default='./net_epoch_196.pth')
    parser.add_argument('--nc', default=20, help="number of channels", type=int)

    opt = parser.parse_args()

    ######################################################################################################################

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if opt.cuda else "cpu")

    ######################################################################################################################
    """Server"""
    HOST = ''
    PORT = 65531
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print('Connected by', addr)

    while True:
        data = conn.recv(16184)
        if not data:
            break
        test_x = pickle.load(data)
        print("test data: ", type(test_x), len(test_x), test_x[0].size)

        """Dataset loading"""
        tensor_test_x = torch.stack([torch.Tensor(i) for i in test_x])  # transform to torch tensors

        tataset = vdata.TensorDataset(tensor_test_x)  # create your datset
        testloader = DataLoader(tataset, drop_last=False)  # create your dataloader

        #######################################################################################################################

        net = Net(insize=opt.len, output_size=128, nc=opt.nc, hidden_size=64, n_layers=2)
        net.apply(weights_init)

        if opt.net != '':
            print("loading trained net...")
            net.load_state_dict(torch.load(opt.net))
        # print(net)

        ratio = _test(opt=opt, net=net, testloader=testloader)

        conn.send(ratio)
    conn.close()

    #######################################################################################################################

if __name__ == '__main__':
    main()
