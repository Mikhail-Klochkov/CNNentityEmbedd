import pathlib, torch

import numpy as np
from torch.functional import F
from typing import Union
from pathlib import Path


def check_path(path: Union[str, pathlib.Path], type='file'):
    if isinstance(path, str):
        path = Path(path)
    if type == 'file':
        if not path.is_file():
            raise FileNotFoundError(f'The file: {path} not found!')
    elif type == 'dir':
        if not path.is_dir():
            raise NotADirectoryError(f'The dir path: {path} not found!')

    return path


def softmax_own(x):
    x_mean = x.mean(dim=-1, keepdims=True)
    x_exp = torch.exp(x-x_mean)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)


def scaled_dot_product(q, k, v, mask=None):
    # BxTxT
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    # BxTxT
    attn_logits /= np.sqrt(q.shape[2])
    if mask is not None:
        # BxTxT first is query second keys (sum by keys)
        attn_logits = attn_logits.masked_fill(mask==0, -9e15) # -inf -> softmax 0
    attention = F.softmax(attn_logits, dim=-1)
    # attention = softmax_own(attn_logits)
    # BxTxT and BxTxd_v -> BxTxd_v
    values = torch.matmul(attention, v)

    return values, attention