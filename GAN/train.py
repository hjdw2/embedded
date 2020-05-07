import numpy as np
import ctypes
import multiprocessing as mp
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from utils import *

real_label = 1
fake_label = 0

def LC_Train(generator_network, discriminator_network, local_discriminator_network, dataloader, args, device):
    optimizerD = optim.Adam(discriminator_network.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD_local = optim.Adam(local_discriminator_network.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(generator_network.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    shm_lists = []
    shape = compute_shapes(generator_network, args)
    shm_loss = SharedTensor([1], dtype='float32')
    shm_shape = SharedTensor(shape)

    shm_lists.append(shm_shape)
    shm_lists.append(shm_shape)
    shm_lists.append(shm_loss)
    shm_lists.append(shm_loss)
    shm_lists.append(shm_shape)

    queue_lists =[]
    for _ in range(0,5):
        queue_lists.append(mp.Queue())

    processes = []
    p = mp.Process(target=train_G, args=(device, dataloader, args, generator_network, optimizerG, shm_lists, queue_lists))
    p.start()
    processes.append(p)
    p = mp.Process(target=train_D, args=(device, dataloader, args, discriminator_network, optimizerD, shm_lists, queue_lists))
    p.start()
    processes.append(p)
    p = mp.Process(target=train_D_LC, args=(device, dataloader, args, local_discriminator_network, optimizerD_local, shm_lists, queue_lists))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

def train_D(device, dataloader, args, model, optimizer, shm_lists, queue_lists):
    model.to(device)
    criterion = nn.BCELoss()
    for epoch in range(args.niter):
        start_time = time.time()
        for i, data in enumerate(dataloader, 0):
            model.zero_grad()
            input = data[0].to(device)
            output = model(input)

            label = torch.full((output.size(0),), real_label, device=device)
            errD_real = criterion(output, label)
            errD_real.backward()

            queue_lists[0].get()
            input = shm_lists[0].recv()
            input = input.to(device)
            output_fake = model(input)

            label.fill_(fake_label)
            errD_fake = criterion(output_fake, label)
            errD_fake.backward()
            optimizer.step()

            errD = errD_real + errD_fake
            shm_lists[2].send(errD_real.data)
            queue_lists[2].put(1)
            shm_lists[3].send(errD_fake.data)
            queue_lists[3].put(1)

            #if i % 5 == 0:
            #    print('[%d/%d][%d/%d]       Loss_D: %.4f '  % (epoch, args.niter, i, len(dataloader), errD.item()))
            #    print('D 1 time: ', time.time()-start_time)
        print('D 1 time: ', time.time()-start_time)

def train_D_LC(device, dataloader, args, model, optimizer, shm_lists, queue_lists):
    model.to(device)
    criterion = nn.BCELoss()
    criterion_mse = nn.L1Loss()
    for epoch in range(args.niter):
        start_time = time.time()
        for i, data in enumerate(dataloader, 0):
            model.zero_grad()
            input = data[0].to(device)
            output = model(input)

            queue_lists[1].get()
            input = shm_lists[1].recv()
            input = input.to(device)
            input = Variable(input, requires_grad=True)

            output_fake = model(input)

            label1 = torch.full((output.size(0),), real_label, device=device)
            errD_real = criterion(output_fake, label1)
            errG = errD_real
            errD_real.backward(retain_graph=True)
            shm_lists[4].send(input.grad.data)
            queue_lists[4].put(1)
            model.zero_grad()

            label2 = torch.full((output.size(0),), real_label, device=device)
            errD_real = criterion(output, label2)
            label3 = torch.full((output.size(0),), fake_label, device=device)
            errD_fake = criterion(output_fake, label3)

            queue_lists[2].get()
            true_errD_real = shm_lists[2].recv()
            queue_lists[3].get()
            true_errD_fake = shm_lists[3].recv()

            true_errD_real = true_errD_real.to(device)
            true_errD_fake = true_errD_fake.to(device)

            loss_lc_real = criterion_mse(errD_real, true_errD_real[0])
            loss_lc_fake = criterion_mse(errD_fake, true_errD_fake[0])
            loss_lc_real.backward()
            loss_lc_fake.backward()
            optimizer.step()

            errD = errD_real + errD_fake

            #if i % 5 == 0:
            #    print('[%d/%d][%d/%d] Loss_D_LC: %.4f Loss_G: %.4f '% (epoch, args.niter, i, len(dataloader), errD.item(), errG.item() ))
            #    print('D 2 time: ', time.time()-start_time)
        print('D LC time: ', time.time()-start_time)


def train_G(device, dataloader, args, model, optimizer, shm_lists, queue_lists):
    criterion = nn.BCELoss()
    model.to(device)
    for epoch in range(args.niter):
        start_time = time.time()
        for i in range(len(dataloader)):
            optimizer.zero_grad()
            noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
            output = model(noise)
            shm_lists[0].send(output.data)
            shm_lists[1].send(output.data)
            queue_lists[0].put(1)
            queue_lists[1].put(1)

            queue_lists[4].get()
            grad = shm_lists[4].recv()
            grad = grad.to(device)

            output.backward(grad)
            optimizer.step()
            time_s = time.time()-start_time

            #if i % 5 == 0:
            #    print('G 1 time: ', time.time()-start_time)

        print('G 1 time: ', time.time()-start_time)
        vutils.save_image(output.detach(), '%s/fake_%03d_%03f.png' % (args.save_image_path, epoch, time_s), normalize=True)
