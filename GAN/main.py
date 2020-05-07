from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import multiprocessing as mp
from models import *
from train import *
from utils import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch GAN Training')
    parser.add_argument('--data_path', type=str, default='data', help='data directory.')
    parser.add_argument('--image_size', type=int, default=64, help='Image size.')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--nz', type=int, default=100, help='noise dimension')
    parser.add_argument('--ngf', type=int, default=128, help='generator dimension')
    parser.add_argument('--ndf', type=int, default=256, help='discriminator dimension')
    parser.add_argument('--nc', type=int, default=3, help='channel dimension')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--no-cuda', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', default=True, help='For Saving the current Model')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')
    parser.add_argument('--save_image_path', type=str, default='images', help='Folder to save generated images.')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')

    parser.add_argument('--time_sleep_iteration', type=int, default=0, help='Time sleep for prevetioning from overhitting CPU or GPU.')

    args = parser.parse_args()

    if not os.path.isdir(args.save_image_path):
        os.mkdir(args.save_image_path)
    
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Model
    print('==> Building model..')
    generator_network = Generator(args)
    discriminator_network = Discriminator(args)
    local_discriminator_network = Discriminator_LC(args)

    LC_Train(generator_network, discriminator_network, local_discriminator_network, dataloader, args, device)

    if args.save_model:
        print('Saving..')
        state = {
            'generator_network': generator_network.state_dict(),
            'discriminator_network': discriminator_network.state_dict(),
            'local_discriminator_network': local_discriminator_network.state_dict(),
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
