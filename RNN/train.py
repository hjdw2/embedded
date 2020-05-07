from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time, math
import multiprocessing as mp
from torch.autograd import Variable
from data import Corpus
from utils import *

def LC_Train(lower_network, upper_network, local_network, train_data, val_data, test_data, args, device, corpus, ntokens):
    shm_lists = []
    shape = compute_shapes(lower_network, train_data, args)
    shm_lists.append(SharedTensor(shape))
    shm_lists.append(SharedTensor([args.nlayers,args.batch_size,args.nhid]))
    shm_lists.append(SharedTensor([args.nlayers,args.batch_size,args.nhid]))
    shm_lists.append(SharedTensor(shape))
    shm_lists.append(SharedTensor([args.batch_size * args.bptt,], dtype='float32'))
    shm_lists.append(SharedTensor(shape))

    queue_lists =[]
    for _ in range(0,4):
        queue_lists.append(mp.Queue())

    processes = []
    p = mp.Process(target=train_lower, args=(lower_network, train_data, test_data, shm_lists, args, queue_lists, corpus, device, ntokens))
    p.start()
    processes.append(p)
    p = mp.Process(target=train_upper, args=(upper_network, train_data, test_data, shm_lists, args, queue_lists, corpus, device, ntokens))
    p.start()
    processes.append(p)
    p = mp.Process(target=train_local, args=(local_network, train_data, test_data, shm_lists, args, queue_lists, corpus, device, ntokens))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

def train_lower(model, train_data, test_data, shm_lists, args, queue_lists, corpus, device, ntokens):
    model.to(device)
    hidden = model.init_hidden(args.batch_size)
    for epoch in range(args.epochs):

        model.train()
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            time.sleep(args.time_sleep_iteration)
            data = get_batch_data(train_data, i, args)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            data = data.to(device)

            model.zero_grad()
            hidden1 = hidden[0].to(device)
            hidden2 = hidden[1].to(device)
            hidden = (hidden1, hidden2)
            output, hidden = model(data, hidden)

            shm_lists[0].send(output.data)
            shm_lists[1].send(hidden[0].data)
            shm_lists[2].send(hidden[1].data)
            shm_lists[3].send(output.data)
            queue_lists[0].put(1)
            queue_lists[1].put(1)

            queue_lists[2].get()
            grad = shm_lists[5].recv()
            grad = grad.to(device)
            output.backward(grad)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-args.lr, p.grad.data)

        model.eval()
        for batch, i in enumerate(range(0, test_data.size(0) - 1, args.bptt)):
            time.sleep(args.time_sleep_iteration)
            data = get_batch_data(test_data, i, args)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            data = data.to(device)
            hidden1 = hidden[0].to(device)
            hidden2 = hidden[1].to(device)
            hidden = (hidden1, hidden2)
            output, hidden = model(data, hidden)

            shm_lists[0].send(output.data)
            shm_lists[1].send(hidden[0].data)
            shm_lists[2].send(hidden[1].data)
            shm_lists[3].send(output.data)
            queue_lists[0].put(1)
            queue_lists[1].put(1)
            queue_lists[2].get()
            queue_lists[3].get()

def train_upper(model, train_data, test_data, shm_lists, args, queue_lists, corpus, device, ntokens):
    model.to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()
    total_loss = 0.

    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        print('Training')
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            time.sleep(args.time_sleep_iteration)
            targets = get_batch_target(train_data, i, args)
            targets = targets.to(device)

            hidden = repackage_hidden(hidden)
            model.zero_grad()

            queue_lists[0].get()
            output = shm_lists[0].recv()
            hidden1 = shm_lists[1].recv()
            hidden2 = shm_lists[2].recv()

            hidden1 = hidden1.to(device)
            hidden2 = hidden2.to(device)
            hidden = (hidden1, hidden2)
            output = output.to(device)

            output, hidden = model(output, hidden)
            loss1 = criterion1(output.view(-1, ntokens), targets)
            loss = criterion2(output.view(-1, ntokens), targets)

            shm_lists[4].send(loss1.data)
            queue_lists[3].put(1)

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-args.lr, p.grad.data)

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('Main| {:5d}/{:5d} batches | ms/batch {:5.2f} | ''loss {:5.2f} | ppl {:8.2f}\n'.format(
                    batch, len(train_data) // args.bptt, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        total_loss = 0
        model.eval()
        print('Test')
        for batch, i in enumerate(range(0, test_data.size(0) - 1, args.bptt)):
            time.sleep(args.time_sleep_iteration)
            targets = get_batch_target(test_data, i, args)
            targets = targets.to(device)
            hidden = repackage_hidden(hidden)

            queue_lists[0].get()
            output = shm_lists[0].recv()
            hidden1 = shm_lists[1].recv()
            hidden2 = shm_lists[2].recv()
            hidden1 = hidden1.to(device)
            hidden2 = hidden2.to(device)
            hidden = (hidden1, hidden2)

            output = output.to(device)

            output, hidden = model(output, hidden)
            loss = criterion2(output.view(-1, ntokens), targets)

            queue_lists[3].put(1)

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('Main| {:5d}/{:5d} batches | ms/batch {:5.2f} | ''loss {:5.2f} | ppl {:8.2f}\n'.format(
                    batch, len(test_data) // args.bptt, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

def train_local(model, train_data, test_data, shm_lists, args, queue_lists, corpus, device, ntokens):
    model.to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    total_loss = 0.
    start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            time.sleep(args.time_sleep_iteration)
            targets = get_batch_target(train_data, i, args)
            targets = targets.to(device)
            model.zero_grad()

            queue_lists[1].get()
            input = shm_lists[3].recv()
            input = input.to(device)
            inputs_var = Variable(input, requires_grad=True)

            output = model(inputs_var)

            loss = criterion2(output.view(-1, ntokens), targets)
            loss.backward(retain_graph=True)
            shm_lists[5].send(inputs_var.grad.data)
            queue_lists[2].put(1)

            queue_lists[3].get()
            loss_true = shm_lists[4].recv()
            loss_true = loss_true.to(device)
            loss_l = criterion1(output.view(-1, ntokens), targets)
            loss_lc = criterion_mse(loss_l, loss_true)

            loss_lc.backward(retain_graph=True)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-args.lr, p.grad.data)

            total_loss += loss.item()
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('Local| {:5d}/{:5d} batches | ms/batch {:5.2f} | ''loss {:5.2f} | ppl {:8.2f}\n'.format(
                    batch, len(train_data) // args.bptt, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        total_loss = 0
        model.eval()
        for batch, i in enumerate(range(0, test_data.size(0) - 1, args.bptt)):
            time.sleep(args.time_sleep_iteration)
            targets = get_batch_target(test_data, i, args)
            targets = targets.to(device)

            queue_lists[1].get()
            input = shm_lists[3].recv()
            input = input.to(device)
            output = model(input)

            loss = criterion2(output.view(-1, ntokens), targets)
            queue_lists[2].put(1)
            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('Local| {:5d}/{:5d} batches | ms/batch {:5.2f} | ''loss {:5.2f} | ppl {:8.2f}\n'.format(
                    batch, len(test_data) // args.bptt, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
