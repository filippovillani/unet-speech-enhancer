import torch
import torch.nn as nn


class SI_SSDR(nn.Module):
    def __init__(self):
        super(SI_SSDR, self).__init__()
    
    def forward(self, s_target, s_hat):
        return self._si_sdr_metric(s_target, s_hat)
    
    def _si_sdr_metric(self,
                       s_hat: torch.Tensor,
                       s_target: torch.Tensor)->torch.Tensor: 
        """
        Compute the Scale-Invariant Signal to Disturb Ratio on the STFT magnitude,
        based on '2013 - A fast griffin-lim algorithm' and on '2018 - SDR - half-baked or well done?'
        Args:
            s_target (torch.Tensor): spectrogram of target signal
            s_hat (torch.Tensor): spectrogram of reconstructed signal
        Returns:
            si_ssdr (float): Scale-Invariant Signal to Disturb Ratio
        """
        # Zero-mean normalization
        s_hat = (s_hat - torch.mean(s_hat))
        s_target = (s_target - torch.mean(s_target))
        
        s_target = torch.div(torch.mul(torch.sum(torch.mul(s_hat, s_target)), s_target),
                            torch.sum(torch.pow(s_target, 2)) + 1e-12)
        
        e_noise = s_hat - s_target
        SI_SSDR_linear = torch.divide(torch.sum(torch.pow(s_target, 2)), torch.sum(torch.pow(e_noise, 2)) + 1e-12)
        si_ssdr = torch.mul(torch.log10(SI_SSDR_linear), 10.)
        return si_ssdr 
    
 
