import os, sys, time
import numpy as np
import random
import torch
from torch.autograd import Variable
import ctypes
import multiprocessing as mp
from collections import deque

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
def get_batch_data(source, i, args):
    seq_len = min(args.bptt, len(source) - 1 - i)
    if seq_len != args.bptt:
        data = source[i+seq_len-args.bptt:i+seq_len]
        return data
    else:
        data = source[i:i+seq_len]
        return data
def get_batch_target(source, i, args):
    seq_len = min(args.bptt, len(source) - 1 - i)
    if seq_len != args.bptt:
        target = source[i+1+seq_len-args.bptt:i+1+seq_len].view(-1)
        return target
    else:
        target = source[i+1:i+1+seq_len].view(-1)
        return target

def shm_as_tensor(mp_array, shape = None):
    '''
    Given a multiprocessing.Array, returns an ndarray pointing to the same data.
    '''
    if mp_array._type_ == ctypes.c_float:
        result = torch.FloatTensor(np.asarray(np.frombuffer(mp_array, dtype=np.float32)))
    elif mp_array._type_ == ctypes.c_long:
        result = torch.LongTensor(np.asarray(np.frombuffer(mp_array, dtype=np.int32)))
    else:
        print('only support float32 or int32')


    if shape is not None:
        result = result.view(*shape)

    return result

def tensor_to_shm(array, data_type='float32', lock = False):
    '''
    Generate an 1D multiprocessing.Array containing the data from the passed ndarray.
    The data will be *copied* into shared memory.
    '''
    array1d = array.view(array.numel())
    if data_type == 'float32':
        c_type = ctypes.c_float
    elif data_type == 'int32':
        c_type = ctypes.c_long
    result = mp.Array(c_type, array.numel(), lock = lock)
    shm_as_tensor(result)[:] = array1d
    return result

class SharedTensor(object):
    def __init__(self, shape, dtype='float32'):
        if dtype == 'float32':
            self.shm_array = tensor_to_shm(torch.zeros(*shape))
        elif dtype == 'int32':
            self.shm_array = tensor_to_shm(torch.LongTensor(*shape).zero_(), data_type='int32')
        else:
            print('only support float32 and int32')
            exit(0)

        if len(shape) > 1:
            self.shm_tensor = shm_as_tensor(self.shm_array, shape=shape)
        else:
            self.shm_tensor = shm_as_tensor(self.shm_array)

        self.inventory = mp.Queue()
        #self.flag = mp.Value('i', 0)

    def recv(self):
        output = self.inventory.get()
        output = output.clone()
        return output

    def send(self, tensor):
        self.shm_tensor[:] = tensor
        self.inventory.put(self.shm_tensor[:])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_shapes(model, inputs, args):
    model.eval()
    hidden = model.init_hidden(args.batch_size)
    output, _ = model(inputs[0:args.bptt], hidden)
    size = output.size()
    shape = [args.bptt,] + [x for x in size[1:]]
    return shape
