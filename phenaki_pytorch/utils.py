import torch
import functools
from datetime import datetime

def format_datetime(input_datetime=datetime.now()):
    formatted_date = input_datetime.strftime("%d/%m/%Y at %H:%M:%S")
    return formatted_date


# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else (val,) * length

def reduce_mult(arr):
    return functools.reduce(lambda x, y: x * y, arr)

def divisible_by(numer, denom):
    return (numer % denom) == 0


# classifier free guidance functions
def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob