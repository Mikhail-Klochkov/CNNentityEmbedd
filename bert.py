import torch, torch.nn as nn

from torch.functional import F

from utils import scaled_dot_product


class MultiHeadAttention(nn.Module):


    def __init__(self, in_dim, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0, 'Incorrect behaviour!'
        # X (B, T, d_in) -> X (B, T, d_in)
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        # use one matrix
        self.QKV = nn.Linear(in_dim, emb_dim * 3)
        self.O_proj = nn.Linear(emb_dim, emb_dim)


    # need xavier initialization Uniform a = (sqrt(6/(f_in + f_out)))
    def _reset(self):
        nn.init.xavier_uniform_(self.QKV.weight)
        self.QKV.bias.data.fill_(0)
        nn.init.xavier_normal_(self.O_proj.weight)
        self.O_proj.bias.data.fill_(0)


    def forward(self, x, mask=None, return_attention=False):
        B, T, d_in = x.size()
        qkv = self.QKV(x)
        # separate Q, K, V for each head
        qkv = qkv.reshape(B, T, self.num_heads, 3*self.head_dim)
        # B x H x T x 3*d_h -> B x H x T x d_h
        qkv = qkv.permute(0,2,1,3)
        # each B x H x T x d_h
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        # B x T x H x d_h
        values = values.permute(0, 2, 1, 3)
        # B x T x H * d_h -> B x T x d_out -> new dimentions
        values = values.reshape(B, T, self.emb_dim)
        # emb_dim x emd_dim values -> B x T x emb_dim
        out = self.O_proj(values)
        if return_attention:
            return out, attention

        return out


class EncoderBlock(nn.Module):


    def __init__(self, in_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.atten = MultiHeadAttention(in_dim, in_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(in_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, in_dim))

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        atten_outs = self.atten(x, mask=mask)
        # dropout + rectified
        x = x + self.dropout(atten_outs)
        # LayerNormalization
        x = self.norm1(x)

        # MLP
        linear_out = self.linear_net(x)
        # dropout + rectified connection
        x = x + self.dropout(linear_out)
        # after attention and linear add LayerNorm
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    # args -> {in_dim, heads, dim_feedforward, dropout=0.0}
    def __init__(self, num_layers, **args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**args) for _ in range(num_layers)])


    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)

        return x


    # mask well be generated in dataloader for all batch
    def get_attention_maps(self, x, mask=None):
        atten_maps = []
        for l in self.layers:
            if not hasattr(l, 'atten'):
                raise ValueError('Incorrect behaviour!')
            _, atten_map = l.atten(x, mask=mask, return_attention=True)
            x = l(x)

        return atten_maps


def test_forward_MultiHeadAttention():
    mheadatt = MultiHeadAttention(100, emb_dim=80, num_heads=8)
    X = torch.randn(8, 72, 100)
    out = mheadatt.forward(X)


if __name__ == '__main__':
    mlhead = MultiHeadAttention(100, 100, 10)
