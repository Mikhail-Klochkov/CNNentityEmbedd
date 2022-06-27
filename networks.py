import torch
import torch.nn as nn
import torch.functional as F

from utils import get_number_torch_model_params


ave_pool = nn.AvgPool1d


class Cnn1dAvePool(nn.Module):


    def __init__(self, in_ch, out_ch, k_size, stride, padding, bias=False, pool_size=2,
                       add_relu=False):
        super().__init__()
        if out_ch % pool_size != 0:
            raise ValueError('Incorrect parameters out_ch and pool_size incorrect!')
        self.layer = [nn.Conv1d(in_ch, out_ch, k_size, stride, padding, bias=bias),
                      ave_pool(pool_size)]
        if add_relu:
            self.layer.append(nn.ReLU())
        self.layer = nn.Sequential(*self.layer)


    def forward(self, x):
        return self.layer(x)



class CNNString(nn.Module):


    def __init__(self, lenght_alphabet,
                       max_lenght,
                       dim_emb,
                       num_channels,
                       times=4,
                       add_relu=False,
                       res_connections=False):
        super().__init__()
        self.len_alphabet = lenght_alphabet
        self.max_len_str = max_lenght
        self.dim_emb = dim_emb
        self.res_connections = res_connections
        self.relu = nn.ReLU()
        # write stack of convolutions with residual conntections
        # need use ModuleDict/ModuleList for self.layers
        self.layers = []
        for idx, t in enumerate(range(times), 1):
            if idx == 1:
                in_ch = lenght_alphabet
            else:
                in_ch = num_channels
            if res_connections:
                # double convolutional layers
                cnnlayer = nn.Sequential(nn.Conv1d(in_ch, num_channels, 3, 1, 1, bias=False),
                                         nn.Conv1d(in_ch, num_channels, 3, 1, 1, bias=False))
                avepool = ave_pool(2)
                self.layers += [('cnn', cnnlayer), ('pool', avepool)]
                # after this add res connection
            else:
                cnnlayer = Cnn1dAvePool(in_ch, num_channels, 3, 1, 1, False, pool_size=2, add_relu=add_relu)
                self.layers += [('cnn', cnnlayer)]
            # --- (B, ch_size, T') --- #
            # conv1d(k=3, s=1, p=1) -> the same shape
            # maybe stacked
            # conv1d(k=3, s=1, p=1) -> the same shape
            # --- (B, ch_size, T') --- #
            # (B, |D|, T) -> (B, ch_size, T // 2) -> (B, ch_size, T // 2 // 2) (not the same lenght)


    def forward(self, x: torch.Tensor):
        for idx, (name_layer, layer) in enumerate(self.layers):
            # after this we add x
            if name_layer == 'cnn':
                if self.res_connections:
                    # ignore first cnn connections
                    if idx == 0:
                        # inside (cnn1, cnn1)
                        x = layer(x)
                    else:
                        x = x + layer(x)
                    x = self.relu(x)
                else:
                    # inside (cnn1, avepool(2), relu(optional))
                    x = layer(x)
            elif name_layer == 'pool':
                # change shape (B, ch_size, T//2)
                x = layer(x)

        return x


class Conv1dBlock(nn.Module):


    def __init__(self, in_channels, out_channels, num_conv=2, pool=None,
                 relu=True, res_conections=True, **convkwargs):
        super().__init__()
        if pool is not None:
            if pool > 2:
                raise ValueError(f'Incorrect pool parameter: "{pool}" > 2')
        if num_conv not in (1, 2):
            raise ValueError(f'Incorrect num_conv parameter: "{num_conv}" not in (1, 2)')

        for k, v in convkwargs.items():
            if k not in ('stride', 'kernel_size', 'padding', 'bias'):
                raise ValueError(f'Incorrect parameter name: {k} with value: {v}.')

        self.res_connections = res_conections
        self.block = nn.ModuleDict(
            {'conv1': nn.Conv1d(in_channels, out_channels, **convkwargs)}
        )
        # add conv2 layers
        if num_conv == 2:
            self.block.update({'conv2': nn.Conv1d(out_channels, out_channels, **convkwargs)})

        if pool is not None:
            self.block.update({'ave_pool': ave_pool(pool)})

        if relu:
            self.block.update({'relu': nn.ReLU()})


    def forward(self, x):
        for k, l in self.block.items():
            if self.res_connections and k == 'conv2':
                # res connections
                if l.in_channels == l.out_channels:
                    x = l(x) + x
            else:
                x = l(x)
        return x


class CNNEmbeddingString(nn.Module):


    def __init__(self, in_ch,
                       out_ch,
                       number_blocks,
                       dim_hidden,
                       dim_emb,
                       res_connections=False,
                       **convoptions):
        super().__init__()
        if dim_hidden > dim_emb:
            raise ValueError(f'Incorrect parameter dim_hidden: {dim_hidden} > {dim_emb}.')
        for k, v in convoptions.items():
            if k not in ('stride', 'kernel_size', 'padding', 'bias'):
                raise ValueError(f'Incorrect parameter name: {k} with value: {v}.')

        self.res_connections = res_connections
        self.dim_emb = dim_emb

        self.embedded = nn.ModuleList()
        self.embedded.append(nn.Conv1d(in_ch, out_ch, **convoptions))
        # number blocks
        for i in range(number_blocks):
            if (i+1) % 2 == 0:
                # in_ch, out_ch, num_conv, pooling, relu, res_connections, other attributes convolution
                block = Conv1dBlock(out_ch, out_ch, 2, 2, True, res_connections, **convoptions)
            else:
                # without pooling
                block = Conv1dBlock(out_ch, out_ch, 2, None, True, res_connections, **convoptions)
            self.embedded.append(block)
        # final part
        self.embedded.append(Conv1dBlock(out_ch, dim_hidden, 1, 2, True, res_connections, **convoptions))
        self.embedded.append(Conv1dBlock(dim_hidden, dim_emb, 1, 2, True, res_connections, **convoptions))


    def forward(self, x):
        # (B, |D|, T_max)
        for idx, layer in enumerate(self.embedded, 1):
            if idx == 1:
                x = layer.forward(x)
            # block
            else:
                x = layer.forward(x)

        return x


if __name__ == '__main__':
    t = torch.randn(10, 26, 28)
    cnnoptions = dict(kernel_size=3, stride=1, padding=1, bias=False)
    cnnembedd = CNNEmbeddingString(in_ch=26, out_ch=20, number_blocks=4,
                                   dim_hidden=30, dim_emb=40,
                                   res_connections=True, **cnnoptions)
    out = cnnembedd(t)
