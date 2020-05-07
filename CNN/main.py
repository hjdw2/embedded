import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import multiprocessing as mp
from models import *
from train import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--data_path', type=str, default='data', help='data directory.')
    parser.add_argument('--image_size', type=int, default=32, help='Image size.')
    parser.add_argument('--image_dim', type=int, default=3, help='Image channel.')

    parser.add_argument('--batch_size', type=int, default=500, help='Batch size.')
    parser.add_argument('--class_num', type=int, default=10, help='For CIFAR10.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--in_channel', type=int, default=3, help='in_channel. 1 for MNIST, 3 for CIFAR10')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
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
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    lower_network = ConvNet_lower(args.in_channel) # nn.Sequential(a,b)
    upper_network = ConvNet_upper(args.class_num) # nn.Sequential(c,d)
    local_network = ConvNet_local(args.class_num)      #

    LC_Train(lower_network, upper_network, local_network, train_loader, test_loader, args, device)

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
