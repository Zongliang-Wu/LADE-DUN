from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .dun_utils import DWPWConvL,DWPWConvTL
import numbers
from .mobileNetV3_cassi import Block as MBlock




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
class RCB(nn.Module):
    def __init__(self, dim=28, n_res=2,act_fn_name="lrelu", bias=False):
        super().__init__()


        self.conv1x1 = BasicConv2d(dim, dim, 1, 1, 1,0, bias=True,act_fn_name="lrelu")
        self.conv3_stack = nn.ModuleList([DWPWConv(dim,dim,3,1,1,bias=bias,act_fn_name="lrelu") for _ in range(n_res)])


    def forward(self, x):
        out = self.conv1x1(x)
        for conv in self.conv3_stack:
            x = conv(x)
            out+=x

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


class SpectralBranch_down(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias",
                 use_codebook=False,
                 use_mask_att=False,
    ):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.use_codebook = use_codebook
        self.use_mask_att = use_mask_att
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
       

        # self.qkv =  DWPWConvL(dim,dim*6,2,2,0)#nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qk =  DWPWConvL(dim,dim*4,2,2,0)#nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_k =  DWPWConvL(dim,dim*2,2,2,0)#nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)

        
        self.spatial_int_down = nn.Conv2d(1,1,2,2,0) 
        self.qkv_dwconv = nn.Conv2d(dim*4, dim*4, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        
        self.qkv_v =  nn.Sequential(nn.Conv2d(dim,dim*2, kernel_size=1, groups=dim),
                                    nn.Conv2d(dim*2, dim*2, kernel_size=(3,5), padding=(1,2),bias=False, groups=dim))
        self.qkv_v_down =  DWPWConvL(dim*2,dim*2,2,2,0)
        # if use_codebook:
        #     self.vq_proj_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias,groups=dim)
        # if use_mask_att:
        #     self.mask_proj_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias,groups=dim)  
            
        self.project_out =  DWPWConvTL(dim*2, dim, 4,2,1)# nn.Conv2d(dim,dim,1,1,0)
        

        

    def forward(self, x, spatial_interaction=None,codebook=None,mask_att=None):# codebook.shape = x.shape
        
        b,c,h,w = x.shape
        x = self.norm(x)
        
        qk = self.qkv_dwconv(self.qk(x))
        q,k = qk.chunk(2, dim=1)
        v = self.qkv_v(x)
        
        if spatial_interaction is not None:
            spatial_interaction_down = self.spatial_int_down(spatial_interaction)
            q = q * spatial_interaction_down
            k = k * spatial_interaction_down
            v = v * spatial_interaction
            

        # if self.use_codebook:
        #     v_code = self.vq_proj_v(codebook)
        #     v = v*v_code
        # if self.use_mask_att:
        #     v_mask_att = self.mask_proj_v(mask_att)
        #     v = v*v_mask_att



        v_down = self.qkv_v_down(v)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_down = rearrange(v_down, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        

        out = (attn @ v_down)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out,v

class SpectralBranch_HIMdown(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias",
                 use_codebook=True,
                 use_mask_att=False,
    ):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.use_codebook = use_codebook
        self.use_mask_att = use_mask_att
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        
        self.norm_z = LayerNorm_Without_Shape(256, LayerNorm_type=LayerNorm_type)

        self.qkv_zl = nn.Linear(256,dim*4)
        self.qkv =  DWPWConvL(dim*2,dim*2,2,2,0)#nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
            
        self.project_out = DWPWConvTL(dim*2, dim, 4,2,1)# nn.Conv2d(dim,dim,1,1,0)
        

        

    def forward(self, x,q, spatial_interaction=None,codebook=None,mask_att=None):# codebook.shape = x.shape
        
        b,c,h,w = x.shape
        
        # q = rearrange(q, 'b head c (w h) -> b (head c) w h', h=h, w=w)
        # q = self.qkv(q)
        kv = self.qkv_zl(codebook)
        k, v = kv.chunk(2, dim=-1) # b  (h_ w

        # q = rearrange(q, 'b (head c) w h -> b head (w h) c', head=self.num_heads)
        q = rearrange(q, 'b (head c) w h -> b head (w h) c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)


        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out,  'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h//2, w=w//2) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out


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


class DW_Inception(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim,
                 bias=False
    ):
        super(DW_Inception, self).__init__()
        # self.branch0 = BasicConv2d(in_dim, out_dim // 4, kernel_size=1, stride=1, bias=bias)

        # self.branch1 = nn.Sequential(
        #     BasicConv2d(in_dim, out_dim // 6, kernel_size=1, stride=1, bias=bias),
        #     BasicConv2d(out_dim // 6, out_dim // 6, kernel_size=3, stride=1, groups=out_dim // 6, padding=1, bias=bias),
        #     BasicConv2d(out_dim // 6, out_dim // 4, kernel_size=1, stride=1, bias=bias)
        # )

        # self.branch2 = nn.Sequential(
        #     BasicConv2d(in_dim, out_dim // 6, kernel_size=1, stride=1, bias=bias),
        #     BasicConv2d(out_dim // 6, out_dim // 6, kernel_size=3, stride=1, groups=out_dim // 6, padding=1, bias=bias),
        #     BasicConv2d(out_dim // 6, out_dim // 4, kernel_size=1, stride=1, bias=bias),
        #     BasicConv2d(out_dim // 4, out_dim // 4, kernel_size=3, stride=1, groups=out_dim // 4, padding=1, bias=bias),
        #     BasicConv2d(out_dim // 4, out_dim // 4, kernel_size=1, stride=1, bias=bias)
        # )

        # self.branch3 = nn.Sequential(
        #     nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
        #     BasicConv2d(in_dim, out_dim//4, kernel_size=1, stride=1, bias=bias)
        # )
        #  Block(3, 24, 88, 24, nn.ReLU, False, 1),
        # kernel_size, in_size, expand_size, out_size, act, se, stride
        # self.mb1 = MBlock(3,out_dim,out_dim//2,out_dim,nn.GELU,False,1)
        # self.mb1 = BasicConv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=bias),
        self.mb = MBlock(3,out_dim,out_dim*2,out_dim,nn.GELU,True,1)

    def forward(self, x):
        # x0 = self.branch0(x)
        # x1 = self.branch1(x)
        # x2 = self.branch2(x)
        # x3 = self.branch3(x)
        # out = torch.cat((x0, x1, x2, x3), 1)

        out = self.mb(x)
        return out


class SpatialBranch(nn.Module):
    def __init__(self,
                 dim,  
                 DW_Expand=2, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)
        self.inception = DW_Inception(
            in_dim=dim, 
            out_dim=dim*DW_Expand, 
            bias=bias
        )
    
    def forward(self, x):
        x = self.norm(x)
        x = self.inception(x)
        return x

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
    
    
class MixS2_Block_split(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 use_codebook = True,
                 use_mask_att = False,
                 return_1 =False,
                 select_codebook=False,
                 use_ch_split = False,
    ):
        super().__init__()
        self.opt = opt
        self.use_codebook = use_codebook
        self.split_ch  = 2 if use_ch_split else 1
        dw_channel = dim//self.split_ch * opt.DW_Expand
        
        if opt.spatial_branch:
            self.spatial_branch = SpatialBranch(
                dim//self.split_ch, 
                DW_Expand=opt.DW_Expand,
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
            self.spatial_gelu = nn.GELU()
            self.spatial_conv = nn.Conv2d(in_channels=dw_channel, out_channels=dim//self.split_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=opt.bias)

        if opt.spatial_interaction:
            self.spatial_interaction = nn.Conv2d(dw_channel, 1, kernel_size=1, bias=opt.bias)

        if opt.spectral_branch:
            self.spectral_branch = SpectralBranch_down(
                opt,
                dim//self.split_ch, 
                num_heads=num_heads, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type,
                use_codebook=use_codebook,
                use_mask_att = use_mask_att,
            )
        if use_codebook:
            self.HIM = SpectralBranch_HIMdown(
                opt,
                dim//self.split_ch, 
                num_heads=num_heads, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type,
                use_codebook=use_codebook,
                use_mask_att = use_mask_att,
            )
            # self.HIM_interaction = nn.Sequential(
            #     nn.Conv2d(dim//self.split_ch, dim // 8, kernel_size=1, bias=opt.bias),
            #     LayerNorm(dim // 8, opt.LayerNorm_type),
            #     nn.GELU(),
            #     nn.Conv2d(dim // 8, dw_channel, kernel_size=1, bias=opt.bias),
            # )

        if opt.spectral_interaction:
            self.spectral_interaction = nn.Sequential(
                nn.Conv2d(dim//self.split_ch, dim // 8, kernel_size=1, bias=opt.bias),
                LayerNorm(dim // 8, opt.LayerNorm_type),
                nn.GELU(),
                nn.Conv2d(dim // 8, dw_channel, kernel_size=1, bias=opt.bias),
            )
            self.spec_to_prior_inter = DWPWConvL(2*dim//self.split_ch, 2*dim//self.split_ch,2,2,0)

        self.add_ss = nn.Sequential(nn.Conv2d(dim,dim,1,1,0),
                                #  nn.GELU(),nn.Conv2d(dim,dim,1,1,0)
                                 )  
        if self.split_ch>1:
            in_dim = 3*dim//self.split_ch
        else:
            in_dim = 2*dim
        self.add_ssp = nn.Sequential(nn.Conv2d(in_dim,dim,1,1,0),
                                #  nn.GELU(),nn.Conv2d(dim,dim,1,1,0)
                                 )
        self.ffn = Residual(
            FFN_FN(
                dim=dim, 
                ffn_name=opt.ffn_name,
                ffn_expansion_factor=opt.FFN_Expand, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
        )
        # if use_codebook and select_codebook:
        #     self.select_codebook = DWPWConvL(dim*2,dim,3,1,1)
        # elif use_codebook:
        #     self.select_codebook = nn.Sequential(DWPWConvL(dim,dim,2,2,0))

        
    def forward(self, x,codebook=None,mask_att=None):
        log_dict = defaultdict(list)
        b, c, h, w = x.shape
        if self.split_ch>1:
            x_spa,x_spec = x.chunk(2, dim=1)
        else:
            x_spa = x
            x_spec = x

        spatial_fea = 0
        spectral_fea = 0
    
        if self.opt.spatial_branch: 
            spatial_identity = x_spa
            spatial_fea = self.spatial_branch(x_spa)
            

        spatial_interaction = None
        if self.opt.spatial_interaction:
            spatial_interaction = self.spatial_interaction(spatial_fea)
            log_dict['block_spatial_interaction'] = spatial_interaction
        
        if self.opt.spatial_branch:
            spatial_fea = self.spatial_gelu(spatial_fea)

        if self.opt.spectral_branch:
           
            spectral_identity = x_spec
            spectral_fea,qkv  = self.spectral_branch(x_spec, spatial_interaction,mask_att=mask_att)
        if self.opt.spectral_interaction:
            spectral_interaction= self.spectral_interaction(
                F.adaptive_avg_pool2d(spectral_fea, output_size=1))
            spectral_interaction = torch.sigmoid(spectral_interaction).tile((1, 1, h, w))
            spatial_fea = spectral_interaction * spatial_fea
            
        if self.use_codebook:
            qkv = self.spec_to_prior_inter(qkv)
            z_fea = self.HIM(x,q=qkv,codebook=codebook)
            # z_interaction = self.HIM_interaction(F.adaptive_avg_pool2d(z_fea, output_size=1))
            # z_interaction = torch.sigmoid(z_interaction).tile((1, 1, h, w))
            # spatial_fea = z_interaction * spatial_fea
            
            
        if self.opt.spatial_branch:
            spatial_fea = self.spatial_conv(spatial_fea)

        if self.opt.spatial_branch:
            spatial_fea = spatial_identity + spatial_fea
            log_dict['block_spatial_fea'] = spatial_fea
        if self.opt.spectral_branch:
            spectral_fea = spectral_identity + spectral_fea
            
            log_dict['block_spectral_fea'] = spectral_fea

        if self.split_ch>1:
            fea_ss = torch.concat([spatial_fea,spectral_fea],dim=1)
        else:
            fea_ss = spatial_fea + spectral_fea
            
        fea_ss = self.add_ss(fea_ss)
        fea = torch.concat([fea_ss,z_fea],dim=1)
        fea = self.add_ssp(fea)
        # fea = torch.concat([spatial_fea,spectral_fea,z_fea],dim=1)
        # fea = self.add(fea)
        out = self.ffn(fea)


        return out, log_dict

class MixS2_Block_split_simple(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 use_codebook = True,
                 use_mask_att = False,
                 return_1 =False,
                 select_codebook=False,
                 use_ch_split = False,
    ):
        super().__init__()
        self.opt = opt
        self.use_codebook = use_codebook
        self.split_ch  = 2 if use_ch_split else 1
        dw_channel = dim//self.split_ch * opt.DW_Expand
        
        if opt.spatial_branch:
            self.spatial_branch = SpatialBranch(
                dim//self.split_ch, 
                DW_Expand=opt.DW_Expand,
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
            self.spatial_gelu = nn.GELU()
            self.spatial_conv = nn.Conv2d(in_channels=dw_channel, out_channels=dim//self.split_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=opt.bias)

        if opt.spatial_interaction:
            self.spatial_interaction = nn.Conv2d(dw_channel, 1, kernel_size=1, bias=opt.bias)

        if opt.spectral_branch:
            self.spectral_branch = SpectralBranch_down(
                opt,
                dim//self.split_ch, 
                num_heads=num_heads, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type,
                use_codebook=use_codebook,
                use_mask_att = use_mask_att,
            )



        if opt.spectral_interaction:
            self.spectral_interaction = nn.Sequential(
                nn.Conv2d(dim//self.split_ch, dim // 8, kernel_size=1, bias=opt.bias),
                LayerNorm(dim // 8, opt.LayerNorm_type),
                nn.GELU(),
                nn.Conv2d(dim // 8, dw_channel, kernel_size=1, bias=opt.bias),
            )
            

            

        self.ffn = Residual(
            FFN_FN(
                dim=dim, 
                ffn_name=opt.ffn_name,
                ffn_expansion_factor=opt.FFN_Expand, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
        )


        
    def forward(self, x,codebook=None,mask_att=None):
        log_dict = defaultdict(list)
        b, c, h, w = x.shape
        if self.split_ch>1:
            x_spa,x_spec = x.chunk(2, dim=1)
        else:
            x_spa = x
            x_spec = x

        spatial_fea = 0
        spectral_fea = 0
    
        if self.opt.spatial_branch: 
            spatial_identity = x_spa
            spatial_fea = self.spatial_branch(x_spa)
            

        spatial_interaction = None
        if self.opt.spatial_interaction:
            spatial_interaction = self.spatial_interaction(spatial_fea)
            log_dict['block_spatial_interaction'] = spatial_interaction
        
        if self.opt.spatial_branch:
            spatial_fea = self.spatial_gelu(spatial_fea)

        if self.opt.spectral_branch:
           
            spectral_identity = x_spec
            spectral_fea,_  = self.spectral_branch(x_spec, spatial_interaction,mask_att=mask_att)
        if self.opt.spectral_interaction:
            spectral_interaction= self.spectral_interaction(
                F.adaptive_avg_pool2d(spectral_fea, output_size=1))
            spectral_interaction = torch.sigmoid(spectral_interaction).tile((1, 1, h, w))
            spatial_fea = spectral_interaction * spatial_fea
            
            
            
        if self.opt.spatial_branch:
            spatial_fea = self.spatial_conv(spatial_fea)

        if self.opt.spatial_branch:
            spatial_fea = spatial_identity + spatial_fea
            log_dict['block_spatial_fea'] = spatial_fea
        if self.opt.spectral_branch:
            spectral_fea = spectral_identity + spectral_fea
            
            log_dict['block_spectral_fea'] = spectral_fea
        if self.split_ch>1:
            fea = torch.cat([spatial_fea,spectral_fea],dim=1)
        else:
            fea = spatial_fea + spectral_fea
        
        out = self.ffn(fea)


        return out, log_dict


class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            # nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.PixelUnshuffle(2),
            # nn.Conv2d(in_channels, in_channels*2, 3, stride=1, padding=1, bias=bias)
            DWPWConvL(in_channels*4, in_channels*2, 3, stride=1, padding=1, bias=bias)
        )
        
        # self.down = pasa_down(in_channels, in_channels*2, 3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            # blinear interpolate may make results not deterministic
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.Conv2d(in_channels, in_channels//2, 3, stride=1, padding=1, bias=bias)
            nn.PixelShuffle(2),
            DWPWConvL(in_channels//4, in_channels//2, 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class MixS2_Transformer_simple(nn.Module):
    def __init__(self, opt,use_codebook=False,use_mask_att=False,first_PM=False):
        super().__init__()
        self.use_codebook = use_codebook
        self.use_mask_att = use_mask_att
        self.opt = opt
        self.embedding = nn.Conv2d(opt.in_dim, opt.dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)
        

                
        self.Encoder = nn.ModuleList([
            
        MixS2_Block_split_simple(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=False),
        MixS2_Block_split_simple(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                          use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=True),
                      
        ])

        self.BottleNeck = MixS2_Block_split_simple(opt = opt, dim = opt.dim * 2 ** 2, num_heads = 2 ** 2,
                                            use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=True)
                      
        self.Decoder = nn.ModuleList([

                    MixS2_Block_split_simple(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                                      use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=True),
              MixS2_Block_split_simple(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=False)
                      
        
        ])
                

        if opt.block_interaction:
            self.BlockInteractions = nn.ModuleList([
                BlockInteraction(opt.dim * 7, opt.dim * 1),
                BlockInteraction(opt.dim * 7, opt.dim * 2)
            ])

        self.Downs = nn.ModuleList([
            DownSample(opt.dim * 2 ** 0, bias=opt.bias),
            DownSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.Ups = nn.ModuleList([
            UpSample(opt.dim * 2 ** 2, bias=opt.bias),
            UpSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 2,
                out_channels = opt.dim * 2 ** 1,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            ),
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 1,
                out_channels = opt.dim * 2 ** 0,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            )
        ])

        if opt.stage_interaction:
            self.stage_interactions = nn.ModuleList([
                StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 2, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias)
            ])


        self.mapping = nn.Conv2d(opt.dim, opt.out_dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)

    def forward(self, x, enc_outputs=None, bottleneck_out=None, dec_outputs=None
                ,codebook=None,mask_att=None):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        enc_outputs_l = []
        dec_outputs_l = []
        x1 = self.embedding(x)

        res1, log_dict1 = self.Encoder[0](x1)

        
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            res1 = self.stage_interactions[0](res1, enc_outputs[0], dec_outputs[0])
        res12 = F.interpolate(res1, scale_factor=0.5, mode='bilinear') 

        x2 = self.Downs[0](res1)

        res2, log_dict2 = self.Encoder[1](x2)

        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            res2 = self.stage_interactions[1](res2, enc_outputs[1], dec_outputs[1])
        res21 = F.interpolate(res2, scale_factor=2, mode='bilinear') 


        x4 = self.Downs[1](res2)
 
        res4, log_dict3 = self.BottleNeck(x4)
 
        
        
        
        if self.opt.stage_interaction and bottleneck_out is not None:
            res4 = self.stage_interactions[2](res4, bottleneck_out, bottleneck_out)

        res42 = F.interpolate(res4, scale_factor=2, mode='bilinear') 
        res41 = F.interpolate(res42, scale_factor=2, mode='bilinear') 
       
        if self.opt.block_interaction:
            res1 = self.BlockInteractions[0](res1, res21, res41) 
            res2 = self.BlockInteractions[1](res12, res2, res42) 
        enc_outputs_l.append(res1)
        enc_outputs_l.append(res2)
        

        dec_res2 = self.Ups[0](res4) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = torch.cat([dec_res2, res2], dim=1) # dim * 2 ** 2
        dec_res2 = self.fusions[0](dec_res2) # dim * 2 ** 2 -> dim * 2 ** 1
 
        dec_res2, log_dict4 = self.Decoder[0](dec_res2)
 
        
        
        
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            dec_res2 = self.stage_interactions[3](dec_res2, enc_outputs[1], dec_outputs[1])
        

        dec_res1 = self.Ups[1](dec_res2) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = torch.cat([dec_res1, res1], dim=1) # dim * 2 ** 1 
        dec_res1 = self.fusions[1](dec_res1) # dim * 2 ** 1 -> dim * 2 ** 0
        
 
        dec_res1, log_dict5 = self.Decoder[1](dec_res1)
 
        
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            dec_res1 = self.stage_interactions[4](dec_res1, enc_outputs[0], dec_outputs[0])

        dec_outputs_l.append(dec_res1)
        dec_outputs_l.append(dec_res2)

        out = self.mapping(dec_res1) + x

        return out[:, :, :h_inp, :w_inp], enc_outputs_l, res4, dec_outputs_l



class MixS2_Transformer(nn.Module):
    def __init__(self, opt,use_codebook=False,use_mask_att=False,first_PM=False):
        super().__init__()
        self.use_codebook = use_codebook
        self.use_mask_att = use_mask_att
        self.opt = opt
        self.embedding = nn.Conv2d(opt.in_dim, opt.dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)
        

                
        self.Encoder = nn.ModuleList([
            
        MixS2_Block_split(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=False),
        MixS2_Block_split(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                          use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=True),
                      
        ])

        self.BottleNeck = MixS2_Block_split(opt = opt, dim = opt.dim * 2 ** 2, num_heads = 2 ** 2,
                                            use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=True)
                      
        self.Decoder = nn.ModuleList([

                    MixS2_Block_split(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                                      use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=True),
              MixS2_Block_split(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_codebook=self.use_codebook,use_mask_att=self.use_mask_att,use_ch_split=False)
                      
        
        ])
                

        if opt.block_interaction:
            self.BlockInteractions = nn.ModuleList([
                BlockInteraction(opt.dim * 7, opt.dim * 1),
                BlockInteraction(opt.dim * 7, opt.dim * 2)
            ])

        self.Downs = nn.ModuleList([
            DownSample(opt.dim * 2 ** 0, bias=opt.bias),
            DownSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.Ups = nn.ModuleList([
            UpSample(opt.dim * 2 ** 2, bias=opt.bias),
            UpSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 2,
                out_channels = opt.dim * 2 ** 1,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            ),
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 1,
                out_channels = opt.dim * 2 ** 0,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            )
        ])

        if opt.stage_interaction:
            self.stage_interactions = nn.ModuleList([
                StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 2, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias)
            ])


        self.mapping = nn.Conv2d(opt.dim, opt.out_dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)

    def forward(self, x, enc_outputs=None, bottleneck_out=None, dec_outputs=None
                ,codebook=None,mask_att=None):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        enc_outputs_l = []
        dec_outputs_l = []
        x1 = self.embedding(x)

        res1, log_dict1 = self.Encoder[0](x1,codebook=codebook[0])

        
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            res1 = self.stage_interactions[0](res1, enc_outputs[0], dec_outputs[0])
        res12 = F.interpolate(res1, scale_factor=0.5, mode='bilinear') 

        x2 = self.Downs[0](res1)

        res2, log_dict2 = self.Encoder[1](x2,codebook=codebook[1])

        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            res2 = self.stage_interactions[1](res2, enc_outputs[1], dec_outputs[1])
        res21 = F.interpolate(res2, scale_factor=2, mode='bilinear') 


        x4 = self.Downs[1](res2)
 
        res4, log_dict3 = self.BottleNeck(x4,codebook=codebook[2])
 
        
        
        
        if self.opt.stage_interaction and bottleneck_out is not None:
            res4 = self.stage_interactions[2](res4, bottleneck_out, bottleneck_out)

        res42 = F.interpolate(res4, scale_factor=2, mode='bilinear') 
        res41 = F.interpolate(res42, scale_factor=2, mode='bilinear') 
       
        if self.opt.block_interaction:
            res1 = self.BlockInteractions[0](res1, res21, res41) 
            res2 = self.BlockInteractions[1](res12, res2, res42) 
        enc_outputs_l.append(res1)
        enc_outputs_l.append(res2)
        

        dec_res2 = self.Ups[0](res4) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = torch.cat([dec_res2, res2], dim=1) # dim * 2 ** 2
        dec_res2 = self.fusions[0](dec_res2) # dim * 2 ** 2 -> dim * 2 ** 1
 
        dec_res2, log_dict4 = self.Decoder[0](dec_res2,codebook=codebook[1])
 
        
        
        
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            dec_res2 = self.stage_interactions[3](dec_res2, enc_outputs[1], dec_outputs[1])
        

        dec_res1 = self.Ups[1](dec_res2) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = torch.cat([dec_res1, res1], dim=1) # dim * 2 ** 1 
        dec_res1 = self.fusions[1](dec_res1) # dim * 2 ** 1 -> dim * 2 ** 0
        
 
        dec_res1, log_dict5 = self.Decoder[1](dec_res1,codebook=codebook[0])
 
        
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            dec_res1 = self.stage_interactions[4](dec_res1, enc_outputs[0], dec_outputs[0])

        dec_outputs_l.append(dec_res1)
        dec_outputs_l.append(dec_res2)

        out = self.mapping(dec_res1) + x

        return out[:, :, :h_inp, :w_inp], enc_outputs_l, res4, dec_outputs_l


def PWDWPWConv(in_channels, out_channels, bias=False, act_fn_name="gelu"):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1, 1, 0, bias=bias),
        nn.GELU(),
        nn.Conv2d(64, 64, 3, 1, 1, bias=bias, groups=64),
        nn.GELU(),
        nn.Conv2d(64, out_channels, 1, 1, 0, bias=bias)
    )

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

# wrong, his version
# def shift_3d(inputs,step=2):
#     [bs, nC, row, col] = inputs.shape
#     for i in range(nC):
#         inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
#     return inputs

# def shift_back_3d(inputs,step=2):
#     [bs, nC, row, col] = inputs.shape
#     for i in range(nC):
#         inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
#     return inputs



def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back_3d(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, nC,row, col] = inputs.shape

    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, i, :,step * i:step * i + col - (nC - 1) * step]
    return output


class DLGD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DL = nn.Sequential(
            PWDWPWConv(29, 28, opt.bias, act_fn_name=opt.act_fn_name),
            PWDWPWConv(28, 28, opt.bias, act_fn_name=opt.act_fn_name),
        )
        # self.r = nn.Parameter(torch.Tensor([0.5]))
        self.minus = nn.Sequential(
            nn.Conv2d(3,1,1,1,0),
        )
        self.DL_grad = nn.Sequential(
            DWPWConvL(56,28,3,1,1),
            DWPWConv(28, 28, 3,1,1,opt.bias, act_fn_name=opt.act_fn_name),
        )


    def forward(self, y, xk_1, Phi):
        """
        y    : (B, 1, 256, 310)
        xk_1 : (B, 28, 256, 310)
        phi  : (B, 28, 256, 310)
        """
        DL_Phi = self.DL(torch.cat([y.unsqueeze(1), Phi], axis=1))
        Phi = DL_Phi + Phi
        phi = A(xk_1, Phi) # (B, 256, 310)
        phixs_init = phi - y
        phixsy = self.minus(torch.concat([phixs_init.unsqueeze(1),phi.unsqueeze(1),y.unsqueeze(1)],axis=1)).squeeze(1)
        phit = At(phixsy, Phi)
        # r = self.r
        xk_1_phit = torch.concat([xk_1, phit], axis=1)
        vk = self.DL_grad(xk_1_phit)

        return vk



from .latent_encoder_arch import latent_encoder_gelu_mobile
  
from .ddpm_denoising_arch import denoising
from .ddpm_denoising_simple_arch import simple_denoise
from .ddpm import DDPM,DDPM_Func
from inspect import isfunction
from einops.layers.torch import Rearrange
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
class DUF_MixS2_diff_HIM4_mlp_abla_full_aggr_smlp_9stg(nn.Module):
    # change encoder conv to mobileblock 1stg: ('32.739 GMac', '2.873 M')
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_phase = opt.train_phase

        if opt.test_mode==True:
            self.gt_le = None
        else:
            self.gt_le = latent_encoder_gelu_mobile()
                
        if hasattr(opt,'apply_ldm'):
            self.apply_ldm = opt.apply_ldm
        else:
            self.apply_ldm = False
        if opt.train_phase==2:
            self.net_le_dm = latent_encoder_gelu_mobile(in_chans=28)
            
            if opt.denoiser=='simple_denoising':
                self.net_d = simple_denoise(64,4,timesteps=opt.timesteps) 
            else:
                self.net_d = denoising(in_channel=256,out_channel=256,
                                       inner_channel=512,block_num=4,group=4,patch_expansion=0.5,channel_expansion=2)
            self.ddpm_func = DDPM_Func()
            
            if self.apply_ldm:
                self.diffusion = DDPM(denoise=self.net_d, 
                                        condition=self.net_le_dm, 
                                        n_feats=64, 
                                        group=4,
                                        linear_start= opt.linear_start,
                                        linear_end= opt.linear_end, 
                                        timesteps = opt.timesteps)
            
        use_codebook_flag = True
        use_mask_flag = False
        # self.res_blocks_long = nn.ModuleList([
        #     RCB(n_res=10) for _ in range(1)])
        # self.add = nn.Sequential(nn.Conv2d(28*2,28,1,1,0,bias=True),nn.GELU())
        # self.res_blocks_short = nn.ModuleList([
        #     RCB(n_res=1) for _ in range(opt.stage)])

        self.head_GD = DLGD(opt)
        self.head_PM = MixS2_Transformer_simple(opt,use_mask_att=use_mask_flag,use_codebook=use_codebook_flag)

        self.body_GD = nn.ModuleList([
            DLGD(opt) for _ in range(opt.stage - 2)
        ]) if not opt.body_share_params else DLGD(opt)
        self.body_PM = nn.ModuleList([
            MixS2_Transformer_simple(opt,use_mask_att=use_mask_flag,use_codebook=use_codebook_flag,) for _ in range(opt.stage - 2)
        ]) if not opt.body_share_params else MixS2_Transformer_simple(opt,use_mask_att=use_mask_flag,use_codebook=use_codebook_flag,)
        self.tail_GD = DLGD(opt)
        self.tail_PM = MixS2_Transformer(opt,use_mask_att=use_mask_flag,use_codebook=use_codebook_flag,)
        m_dim = 28*4
        group = 4
        embed_dim = 64
        self.down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, 1),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
    def forward(self, y, Phi,gt=None):

        log_dict = defaultdict(list)
        B, C, H, W = Phi.shape
        x0 = y.unsqueeze(1).repeat((1, C, 1, 1))
        phi_s = torch.sum(Phi,1,keepdim=True)       
        inp_img = (x0/phi_s)*Phi
        inp_img  = shift_back_3d(inp_img)
        
        if gt is not None:
            prior_z = self.gt_le(inp_img,gt)
            
        else:
            prior_z = None
        # else:
        #     x0 = y.unsqueeze(1).repeat((1, C, 1, 1))
            
                    # freeze c1 (cpen_s1)

        if self.train_phase==1:
            prior = prior_z
        elif self.train_phase==2:

            if self.apply_ldm:
                
                prior, _=self.diffusion(inp_img,prior_z)
            else:
                prior_d = self.net_le_dm(inp_img)
                # diffusion-forward
                t = self.opt.timesteps #self.opt['diffusion_schedule']['timesteps']
                # [b, 4c']
                noise = default(noise, lambda: torch.randn_like(prior_z))
                # sample xt/x_noisy (from x0/x_start)
                prior_noisy = self.ddpm_func.q_sample(
                    x_start=prior_z, sqrt_alpha_cumprod=self.alphas_cumprod[t-1],
                    noise=noise)
                # diffusion-reverse

                prior = self.ddpm_func.p_sample_loop_wo_variance(prior_d, prior_noisy,net_d_noise=self.net_d)



            log_dict['prior'] = prior
            log_dict['prior_z'] = prior_z
            
        if self.train_phase>0:
            prior_att = []

            prior_1 = prior # [2, 16, 256]
            prior_2 = self.down_1(prior_1)# [2, 4, 256]
            prior_3 = self.down_2(prior_2) # [2, 256]
            
            prior_att.append(prior_1)
            prior_att.append(prior_2) 
            prior_att.append(prior_3) #112 64 64
        elif self.train_phase==0:
            prior_att = [None,None,None]
        ##############

        B, C, H, W = Phi.shape    
            
        
        # x_pre_res = self.res_blocks_long[0](inp_img)   
        v_pre = self.head_GD(y, x0, Phi)
        v_pre = shift_back_3d(v_pre)
        # log_dict['stage0_v'] = v_pre
        x_pre, enc_outputs, bottolenect_out, dec_outputs = self.head_PM(v_pre,codebook=prior_att,)
        log_dict['stage0_x'] = x_pre
        x_pre = shift_3d(x_pre)
        
       

        for i in range(self.opt.stage-2):
            v_pre = self.body_GD[i](y, x_pre, Phi) if not self.opt.body_share_params else self.body_GD(y, x_pre, Phi)
            v_pre = shift_back_3d(v_pre)
            # log_dict[f'stage{i+1}_v'] = v_pre
            x_pre, enc_outputs, bottolenect_out, dec_outputs = \
                self.body_PM[i](v_pre, enc_outputs, bottolenect_out, dec_outputs,codebook=prior_att,) if not self.opt.body_share_params else self.body_PM(v_pre, enc_outputs, bottolenect_out, dec_outputs,codebook=prior_att,)
            log_dict[f'stage{i+1}_x'] = x_pre
            x_pre = shift_3d(x_pre)
            
        
        v_pre = self.tail_GD(y, x_pre, Phi)
        v_pre = shift_back_3d(v_pre)
        # log_dict[f'stage{self.opt.stage-1}_v'] = v_pre
        out, enc_outputs, bottolenect_out, dec_outputs = self.tail_PM(v_pre, 
                                                                      enc_outputs, bottolenect_out, dec_outputs,codebook=prior_att,)
        # log_dict[f'stage{self.opt.stage-1}_x'] = out
    
        out = out[:, :, :, :256]

        return out, log_dict