import torch
import torch.nn as nn

from .layers import PInvBlock, ConvBlock


class PInvDAE(nn.Module):
    def __init__(self, hparams):

        super(PInvDAE, self).__init__()
        self.device = hparams.device

        out_channels = hparams.conv_channels + hparams.conv_channels[-1::-1] + [1]
        in_channels = out_channels[::-1]
        in_channels = in_channels[:((len(out_channels)//2 + 1))] + [x*2 for x in out_channels[(len(out_channels)//2):-1]]
        
        
        self.pinvblock = PInvBlock(hparams)
        self.convblocks = nn.ModuleList([ConvBlock(in_channels[l], 
                                                   out_channels[l], 
                                                   hparams.conv_kernel_size,
                                                   hparams.drop_rate) for l in range(len(in_channels))])
        self.convblocks[-1].drop_rate = 0.
        self.n_blocks = len(self.convblocks)
        
    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]
        Returns:
            _type_: _description_
        """
        x = self.pinvblock(melspec)

        # Encoder
        x_cat = []
        for l in range(self.n_blocks//2):
            x = self.convblocks[l](x)
            x_cat.append(x)
        x_cat = x_cat[::-1]
        
        x = self.convblocks[self.n_blocks//2](x)
        # Decoder
        for l in range(self.n_blocks//2+1, self.n_blocks):
            x = torch.cat([x, x_cat[l-(self.n_blocks//2+1)]], axis=1)
            x = self.convblocks[l](x)

        x_max = torch.as_tensor([torch.max(x[q]) for q in range(x.shape[0])])
        stft_hat = torch.empty(x.shape)
        for b in range(stft_hat.shape[0]):
            stft_hat[b] = x[b] / x_max[b]
        stft_hat = stft_hat.to(x.device)
        return stft_hat