import torch
import torch.nn as nn
import torch.functional as F


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


if __name__ == '__main__':
    t = torch.randn(10, 26, 28)
    cnnembedd =  CNNString(lenght_alphabet=26, max_lenght=28, dim_emb=50, num_channels=8, times=4)
