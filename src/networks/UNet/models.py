import torch.nn as nn

from .layers import ContractingBlock, Bottleneck, ExpandingBlock, OutBlock

class UNet(nn.Module):
    def __init__(self, hparams):
        
        super(UNet, self).__init__()
        self.device = hparams.device
        self.n_unet_blocks = hparams.n_unet_blocks
        
        self.encoder = nn.ModuleList([ContractingBlock(in_channels = hparams.n_channels,
                                                       out_channels = hparams.first_unet_channel_units,
                                                       kernel_size = hparams.unet_kernel_size,
                                                       drop_rate = hparams.drop_rate)])
        
        for b in range(self.n_unet_blocks-1):
            self.encoder.append(ContractingBlock(in_channels = 2**b * hparams.first_unet_channel_units,
                                                 kernel_size = hparams.unet_kernel_size,
                                                 drop_rate = hparams.drop_rate))
        
        self.bottleneck = Bottleneck(in_channels = (2**(b+1)) * hparams.first_unet_channel_units,
                                                    kernel_size = hparams.unet_kernel_size,
                                                    drop_rate = hparams.drop_rate)
        
        self.decoder = nn.ModuleList([ExpandingBlock(in_channels = 2**b * hparams.first_unet_channel_units,
                                                     kernel_size = hparams.unet_kernel_size,
                                                     drop_rate = hparams.drop_rate) for b in range(self.n_unet_blocks,0,-1)])
        
        self.outblock = OutBlock(in_channels = hparams.first_unet_channel_units)
        
        
    def forward(self, x):
        
        x_cat = []
        for block in self.encoder:
            x, x_cat_ = block(x)
            x_cat.append(x_cat_)
        x_cat = x_cat[-1::-1]    
        
        x = self.bottleneck(x)
        
        for b,block in enumerate(self.decoder):
            x = block(x, x_cat[b])

        out = self.outblock(x)
        
        return out  