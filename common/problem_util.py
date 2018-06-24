import os


GPU_INDEX = 0
Parallel = False


def to_cuda(x):
    if not Parallel:
        return x.cuda(GPU_INDEX)
    else:
        return x.cuda()


def get_gpu_index():
    return GPU_INDEX


PAD_VALUE = -1
