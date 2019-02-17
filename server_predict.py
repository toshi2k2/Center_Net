from __future__ import print_function

# import os
# import random
# import math
import torch
from torch.autograd import Variable
import numpy as np

def _test(opt, net, testloader):
    # device = torch.device("cuda" if opt.cuda else "cpu")
    if opt.cuda:
        net.cuda()
    # setup optimizer
    print("Testing...\n")

    total, nonzero = 0, 0

    for images in testloader:
        if opt.cuda:
            images = images.cuda()
        images = Variable(images)
        outputs = net(images)
        predicted = torch.argmax(outputs, dim=1)
        if opt.cuda:
            p2 = predicted.cpu().detach().numpy()
        else:
            p2 = predicted.numpy()
        nonzero += np.count_nonzero(p2)
        total += p2.shape[0]

    return ((total-nonzero)/total)*100

