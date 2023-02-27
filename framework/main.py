import sys
import math
import torch
import ctypes
import datetime
import numpy as np
import argparse
import time
import random
import os
from model import TGCtrain

FType = torch.FloatTensor
LType = torch.LongTensor


def main_train(args):
    start = datetime.datetime.now()
    the_train = TGCtrain.TGC(args)
    the_train.train()
    end = datetime.datetime.now()
    print('Training Complete with Time: %s' % str(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='arxivAI')
    parser.add_argument('--clusters', type=int, default=5)
    # dblp/10, arxivAI/5
    parser.add_argument('--epoch', type=int, default=200)
    # dblp/50, arxivAI/200
    parser.add_argument('--neg_size', type=int, default=2)
    parser.add_argument('--hist_len', type=int, default=1)
    # dblp/5, arxivAI/1
    parser.add_argument('--save_step', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--directed', type=bool, default=False)
    args = parser.parse_args()

    main_train(args)