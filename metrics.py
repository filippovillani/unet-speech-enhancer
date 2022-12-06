import torch

def si_nsr_loss(enhanced_speech: torch.Tensor, 
                clean_speech: torch.Tensor)->torch.Tensor:

    s_hat = enhanced_speech - torch.mean(enhanced_speech)
    s_target = clean_speech - torch.mean(clean_speech) 
       
    s_target = torch.div(torch.mul(torch.sum(torch.mul(s_hat, s_target)), s_target),
                         torch.sum(torch.pow(s_target, 2)) + 1e-12)
    
    e_noise = s_hat - s_target
    SI_NSR_linear = torch.divide(torch.sum(torch.pow(e_noise, 2)), torch.sum(torch.pow(s_target, 2)))
    SI_NSR = torch.mul(torch.log10(SI_NSR_linear), 10.)
    return SI_NSR
  
def si_snr_metric(enhanced_speech: torch.Tensor, 
                  clean_speech: torch.Tensor)->torch.Tensor:
    
    s_hat = enhanced_speech - torch.mean(enhanced_speech)
    s_target = clean_speech - torch.mean(clean_speech) 
       
    s_target = torch.div(torch.mul(torch.sum(torch.mul(s_hat, s_target)), s_target),
                         torch.sum(torch.pow(s_target, 2)) + 1e-12)
    
    e_noise = s_hat - s_target
    SI_SNR_linear = torch.divide(torch.sum(torch.pow(s_target, 2)), torch.sum(torch.pow(e_noise, 2)))
    SI_SNR = torch.mul(torch.log10(SI_SNR_linear), 10.)
    return SI_SNR   
 
