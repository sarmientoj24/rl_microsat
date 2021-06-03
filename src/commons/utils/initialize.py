import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)