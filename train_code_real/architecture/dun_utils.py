

import torch
import torch.nn as nn

from einops import rearrange

import numbers



def A(x, Phi):
    B, nC, H, W = x.shape
    temp = x * Phi
    y = torch.sum(temp, 1)
    y = y / nC * 2
    return y

def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x





def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back_3d(inputs, step=2):   
    [bs, nC,row, col] = inputs.shape

    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, i, :,step * i:step * i + col - (nC - 1) * step]
    return output



class BasicConv2d(nn.Module):
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 kernel_size, 
                 stride, 
                 groups = 1, 
                 padding = 0, 
                 bias = False,
                 act_fn_name = "gelu",
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act_fn(x)
        return x


def DWConv(dim, kernel_size, stride, padding, bias=False):
    return nn.Conv2d(dim, dim, kernel_size, stride, padding, bias=bias, groups=dim)

def PWConv(in_dim, out_dim, bias=False):
    return nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias)

def DWPWConv(in_dim, out_dim, kernel_size, stride, padding, bias=False, act_fn_name="gelu"):
    return nn.Sequential(
        DWConv(in_dim, kernel_size, stride, padding, bias),
        nn.GELU(),
        PWConv(in_dim, out_dim, bias)
    )
def DWConvT(dim, kernel_size, stride, padding, bias=False):
    return nn.ConvTranspose2d(dim, dim, kernel_size, stride, padding, bias=bias, groups=dim)

def DWPWConvL(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        # DefConv(in_dim, in_dim, kernel_size, stride, padding, bias=bias, groups=in_dim),
        DWConv(in_dim, kernel_size, stride, padding, bias),
        PWConv(in_dim, out_dim, bias)
    )
    
def DWPWConvTL(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        # DefConv(in_dim, in_dim, kernel_size, stride, padding, bias=bias, groups=in_dim),
        DWConvT(in_dim, kernel_size, stride, padding, bias),
        PWConv(in_dim, out_dim, bias)
    )
def PWDWPWConv(in_channels, out_channels, bias=False, act_fn_name="gelu"):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1, 1, 0, bias=bias),
        nn.GELU(),
        nn.Conv2d(64, 64, 3, 1, 1, bias=bias, groups=64),
        nn.GELU(),
        nn.Conv2d(64, out_channels, 1, 1, 0, bias=bias)
    )  
    
class BlockInteraction(nn.Module):
    def __init__(self, in_channel, out_channel, act_fn_name="gelu", bias=False):
        super(BlockInteraction, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=bias)
        )
       
    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)




class StageInteraction(nn.Module):
    def __init__(self, dim, act_fn_name="lrelu", bias=False):
        super().__init__()
        self.st_inter_enc = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.st_inter_dec = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.act_fn = nn.GELU()
        self.phi = DWConv(dim, 3, 1, 1, bias=bias)
        self.gamma = DWConv(dim, 3, 1, 1, bias=bias)

    def forward(self, inp, pre_enc, pre_dec):
        out = self.st_inter_enc(pre_enc) + self.st_inter_dec(pre_dec)
        skip = self.act_fn(out)
        phi = torch.sigmoid(self.phi(skip))
        gamma = self.gamma(skip)

        out = phi * inp + gamma

        return out

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        return x + self.module(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66, 
                 bias = False,
                 LayerNorm_type = "WithBias",
                 act_fn_name = "gelu"
    ):
        super(Gated_Dconv_FeedForward, self).__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x


def FFN_FN(
    ffn_name,
    dim, 
    ffn_expansion_factor=2.66, 
    bias=False,
    LayerNorm_type="WithBias",
    act_fn_name = "gelu"
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim, 
                ffn_expansion_factor=ffn_expansion_factor, 
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                act_fn_name = act_fn_name
            )
# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            
            nn.PixelUnshuffle(2),
            
            DWPWConvL(in_channels*4, in_channels*2, 3, stride=1, padding=1, bias=bias)
        )
        
       

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            DWPWConvL(in_channels//4, in_channels//2, 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x = self.up(x)
        return x

