import os


GPU_INDEX = 0
Parallel = False


def to_cuda(x):
    if Parallel:
        return x.cuda()
    elif GPU_INDEX is None:
        return x.cpu()
    elif not Parallel:
        return x.cuda(GPU_INDEX)


def get_gpu_index():
    return GPU_INDEX


PAD_VALUE = -1
