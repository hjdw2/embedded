from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import multiprocessing as mp
from torch.autograd import Variable
from utils import *

def LC_Train(lower_network, upper_network, local_network, train_loader, test_loader, args, device):
    optimizer_1 = torch.optim.Adam(lower_network.parameters(), lr=args.lr)
    optimizer_2 = torch.optim.Adam(upper_network.parameters(), lr=args.lr)
    optimizer_LC = torch.optim.Adam(local_network.parameters(), lr=args.lr)
    shm_lists = []
    shm_target = SharedTensor([args.batch_size,], dtype='int32')
    shm_lists.append(shm_target) # [0]
    shm_lists.append(shm_target) # [1]

    test_inputs = torch.FloatTensor(1, args.image_dim, args.image_size, args.image_size) # for CIFAR10, for MNIST (1,1,28,28)
    shape = compute_shapes(lower_network, test_inputs, args)
    shm_data = SharedTensor(shape)
    shm_lists.append(shm_data) # [2]
    shm_lists.append(shm_data) # [3]
    shm_loss = SharedTensor([args.batch_size, ], dtype='float32')
    shm_lists.append(shm_loss) # [4]
    shm_lists.append(shm_data) # [5]

    queue_lists =[]
    for _ in range(0,5):
        queue_lists.append(mp.Queue())

    print('              Epoch   Train_Acc(%)   Train_Loss   Test_Acc(%)   Test_Loss   Training_time(s)')
    processes = []
    p = mp.Process(target=train_lower, args=(train_loader, test_loader, lower_network, optimizer_1, shm_lists, args, queue_lists, device))
    p.start()
    processes.append(p)
    p = mp.Process(target=train_upper, args=(train_loader, test_loader, upper_network, optimizer_2, shm_lists, args, queue_lists, device))
    p.start()
    processes.append(p)
    p = mp.Process(target=train_local, args=(train_loader, test_loader, local_network, optimizer_LC, shm_lists, args, queue_lists, device))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

def train_lower(train_loader, test_loader, model, optimizer, shm_lists, args, queue_lists, device):
    model.to(device)
    for epoch in range(args.epochs):
        model.train()
        for i, (inputs, target) in enumerate(train_loader):
            while len(inputs) != args.batch_size:
                inputs_copy_len = (args.batch_size - len(inputs)) if (args.batch_size - len(inputs) <= len(inputs)) else len(inputs)
                inputs = torch.cat([inputs, inputs[0:inputs_copy_len]], 0)
                target = torch.cat([target, target[0:inputs_copy_len]], 0)
            time.sleep(args.time_sleep_iteration)
            # send target to the last processor
            inputs = inputs.to(device)
            output = model.forward(inputs)

            shm_lists[0].send(target)
            shm_lists[1].send(target)
            shm_lists[2].send(output.data)
            shm_lists[3].send(output.data)
            queue_lists[1].put(1)
            queue_lists[2].put(1)

            optimizer.zero_grad()
            queue_lists[4].get()
            grad = shm_lists[5].recv()
            grad = grad.to(device)

            output.backward(grad)
            optimizer.step()

        model.eval()
        for i, (inputs, target) in enumerate(test_loader):
            while len(inputs) != args.batch_size:
                inputs_copy_len = (args.batch_size - len(inputs)) if (args.batch_size - len(inputs) <= len(inputs)) else len(inputs)
                inputs = torch.cat([inputs, inputs[0:inputs_copy_len]], 0)
                target = torch.cat([target, target[0:inputs_copy_len]], 0)
            time.sleep(args.time_sleep_iteration)

            # send target to the last processor
            inputs = inputs.to(device)
            output = model.forward(inputs)

            shm_lists[0].send(target)
            shm_lists[1].send(target)
            shm_lists[2].send(output.data)
            shm_lists[3].send(output.data)
            queue_lists[1].put(1)
            queue_lists[2].put(1)
            queue_lists[3].get()
            queue_lists[4].get()

def train_upper(train_loader, test_loader, model, optimizer, shm_lists, args, queue_lists, device):
    model.to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_losses = AverageMeter()
        train_acc = AverageMeter()
        test_losses = AverageMeter()
        test_acc = AverageMeter()

        for i in range(len(train_loader)):
            time.sleep(args.time_sleep_iteration)

            queue_lists[1].get()
            target = shm_lists[0].recv()
            target = target.to(device)
            inputs = shm_lists[2].recv()
            inputs = inputs.to(device)
            target_var = Variable(target)
            inputs_var = Variable(inputs, requires_grad=True)

            output = model(inputs_var)
            loss1 = criterion1(output, target_var)
            loss = criterion2(output, target_var)
            shm_lists[4].send(loss1.data)
            queue_lists[3].put(1)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            train_losses.update(loss.data, target.size(0))
            train_acc.update(prec1, target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time = time.time()-start_time
        for i in range(len(test_loader)):
            time.sleep(args.time_sleep_iteration)

            queue_lists[1].get()
            target = shm_lists[0].recv()
            inputs = shm_lists[2].recv()
            target = target.to(device)
            inputs = inputs.to(device)

            output = model(inputs)
            loss = criterion2(output, target)
            queue_lists[3].put(1)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            test_losses.update(loss.data, target.size(0))
            test_acc.update(prec1, target.size(0))

        print('Main Network ','{epoch:d}        {acc.avg:.3f}         {losses.avg:.3f}        {test_acc.avg:.3f}        {test_losses.avg:.3f}       {time:.3f}  \n'.format(epoch=epoch, acc=train_acc, losses=train_losses, test_acc=test_acc, test_losses=test_losses, time=training_time), end=' ',flush=True)

def train_local(train_loader, test_loader, model, optimizer, shm_lists, args, queue_lists, device):
    model.to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_losses = AverageMeter()
        train_acc = AverageMeter()
        test_losses = AverageMeter()
        test_acc = AverageMeter()
        for i in range(len(train_loader)):
            time.sleep(args.time_sleep_iteration)

            queue_lists[2].get()
            target = shm_lists[1].recv()
            inputs = shm_lists[3].recv()
            inputs = inputs.to(device)
            target = target.to(device)
            target_var = Variable(target)
            inputs_var = Variable(inputs, requires_grad=True)

            output = model(inputs_var)
            loss = criterion2(output, target_var)
            loss.backward(retain_graph=True)
            shm_lists[5].send(inputs_var.grad.data)
            queue_lists[4].put(1)

            queue_lists[3].get()
            loss_true = shm_lists[4].recv()
            loss_true = loss_true.to(device)

            loss_l = criterion1(output, target_var)
            loss_lc = criterion_mse(loss_l, loss_true)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            train_losses.update(loss.data, target.size(0))
            train_acc.update(prec1, target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_lc.backward(retain_graph=True)
            optimizer.step()

        training_time = time.time()-start_time
        for i in range(len(test_loader)):
            time.sleep(args.time_sleep_iteration)


            queue_lists[2].get()
            target = shm_lists[1].recv()
            inputs = shm_lists[3].recv()
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            loss = criterion2(output, target)
            queue_lists[4].put(1)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            test_losses.update(loss.data, target.size(0))
            test_acc.update(prec1, target.size(0))

        print('Local Network ','{epoch:d}        {acc.avg:.3f}         {losses.avg:.3f}        {test_acc.avg:.3f}        {test_losses.avg:.3f}       {time:.3f}  \n'.format(epoch=epoch, acc=train_acc, losses=train_losses, test_acc=test_acc, test_losses=test_losses, time=training_time), end=' ',flush=True)
