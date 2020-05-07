import numpy as np
import ctypes
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf ),
            nn.ReLU(True),
            # state size. (args.ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf ),
            nn.ReLU(True),
            # state size. (args.ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf ),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (args.ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (args.nc) x 64 x 64
        )
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (args.nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf, args.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf*4) x 8 x 8
            nn.Conv2d(args.ndf, args.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf*8) x 4 x 4
            nn.Conv2d(args.ndf, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

class Discriminator_LC(nn.Module):
    def __init__(self, args):
        super(Discriminator_LC, self).__init__()
        self.main = nn.Sequential(
            # state size. (args.ndf*2) x 16 x 16
            nn.Conv2d(args.nc, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf) x 32 x 32
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
