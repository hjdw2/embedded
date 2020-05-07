from __future__ import print_function
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import multiprocessing as mp
from collections import deque
from data import Corpus
from utils import *
from models import *
from train import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN Language Model')
    parser.add_argument('--data', type=str, default='input', help='location of the data corpus')
    parser.add_argument('--epochs', type=int, default=10,      help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')

    parser.add_argument('--lr', type=float, default=20,       help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,   help='gradient clipping')
    parser.add_argument('--emsize', type=int, default=128,    help='size of word embeddings')
    parser.add_argument('--nlayers', type=int, default=1,     help='number of layers')
    parser.add_argument('--nhid', type=int, default=1500,     help='number of hidden units per layer')
    parser.add_argument('--bptt', type=int, default=35,       help='sequence length')

    parser.add_argument('--log-interval', type=int, default=100, help='report interval')
    parser.add_argument('--seed', type=int, default=1,        help='random seed')
    parser.add_argument('--no-cuda', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', default=True, help='For Saving the current Model')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')

    parser.add_argument('--time_sleep_iteration', type=int, default=0, help='Time sleep for prevetioning from overhitting CPU or GPU.')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data
    print('==> Preparing data..')
    corpus = Corpus('input')
    ntokens = len(corpus.dictionary)

    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, args.batch_size)
    test_data = batchify(corpus.test, args.batch_size)

    # Model
    print('==> Building model..')
    lower_network = RNN_lower('LSTM', ntokens, args.emsize, args.nhid, args.nlayers)
    upper_network = RNN_upper('LSTM', ntokens, args.nhid, args.nhid, args.nlayers)
    local_network = RNN_local(ntokens, args.nhid, args.nhid, args.nlayers)

    LC_Train(lower_network, upper_network, local_network, train_data, val_data, test_data, args, device,corpus, ntokens)

    if args.save_model:
        print('Saving..')
        state = {
            'lower_network': lower_network.state_dict(),
            'upper_network': upper_network.state_dict(),
            'local_network': local_network.state_dict(),
        }
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        torch.save(state, args.save_path+'/ckpt.pth')

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
