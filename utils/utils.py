# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import random
import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def reduce_tensors(tensor):
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    return reduced_tensor


def average_gradients(model):
    """ average gradients """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)
