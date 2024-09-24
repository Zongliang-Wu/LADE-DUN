import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

        return x



    



from .mobileNetV3_cassi import Block as MBlock
class latent_encoder_gelu_mobile(nn.Module):

    def __init__(self, in_chans=56, embed_dim=64, block_num=2, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(latent_encoder_gelu_mobile, self).__init__()


        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = MBlock(3,in_chans*16,in_chans*4,embed_dim,nn.GELU,True,1)
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                MBlock(3,embed_dim,embed_dim*4,embed_dim,nn.GELU,True,1))

                
            self.blocks.append(block)

        self.conv2 = MBlock(3,embed_dim,embed_dim*4,embed_dim,nn.GELU,True,1) 

        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = MLP(num_patches=group*group, embed_dims=embed_dim, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        # 33% params
        self.end = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*4),
                nn.GELU(),)
        

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.pool(self.conv2(x))

        x = rearrange(x, 'b c h w-> b (h w) c') # [2, 64, 4, 4] to 
        x = self.mlp(x)
        x = self.end(x)
        return x # [2, wh, 64*4]

