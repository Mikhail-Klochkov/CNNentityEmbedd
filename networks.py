import torch
import torch.nn as nn
import torch.functional as F


class CNNString(nn.Module):


    def __init__(self, alpha_len, max_len_string, d_emb, channel, mtc_input):
        super(CNNString, self).__init__()
        self.alpha_len = alpha_len
        self.max_len_string = max_len_string
        self.d_emb = d_emb

        # init multi_CNN_layer

