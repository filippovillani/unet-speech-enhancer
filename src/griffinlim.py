import torch
import numpy as torch
import torch


def griffin_lim(spectrogram: torch.Tensor, 
                n_iter: int = 500, 
                n_fft: int = 1024,
                init: str = "zeros"):
    
    X_init_phase = initialize_phase(spectrogram, init)
    
    X = spectrogram * torch.exp(1j * X_init_phase)
    for _ in range(n_iter):
        X_hat = torch.istft(X, 
                            n_fft = n_fft,
                            window = torch.hann_window(1024).to(X.device))    # G+ cn
        X_hat = torch.stft(X_hat, 
                           n_fft=n_fft, 
                           return_complex = True,
                           window = torch.hann_window(n_fft).to(X_hat.device)) # G G+ cn  
        X_phase = torch.angle(X_hat) 
        X = spectrogram * torch.exp(1j * X_phase)   # Pc1(Pc2(cn-1))  
    
    x = torch.istft(X, 
                    n_fft = n_fft,
                    window = torch.hann_window(1024).to(X.device))
    
    return x


def fast_griffin_lim(spectrogram: torch.Tensor,
                     n_fft: int = 1024,
                     n_iter: int = 500,
                     alpha: float = 0.99, 
                     init: str = "zeros"):
    
    spectrogram = spectrogram.squeeze()
    # Initialize the algorithm
    if spectrogram.dtype not in [torch.complex64, torch.complex128]:
        X_init_phase = initialize_phase(spectrogram, init)
        X = spectrogram * torch.exp(1j * X_init_phase)
    else:
        X = spectrogram

    prev_proj = torch.istft(X, 
                            n_fft = n_fft,
                            window = torch.hann_window(n_fft).to(X.device))
    prev_proj = torch.stft(prev_proj, 
                           n_fft=n_fft, 
                           window = torch.hann_window(n_fft).to(prev_proj.device),
                           return_complex = True)

    prev_proj_phase = torch.angle(prev_proj) 
    prev_proj = spectrogram * torch.exp(1j * prev_proj_phase) 
    
    for _ in range(n_iter+1):
        curr_proj = torch.istft(X, 
                                n_fft = n_fft,
                                window = torch.hann_window(n_fft).to(X.device))    # G+ cn            
        curr_proj = torch.stft(curr_proj, 
                               n_fft=n_fft, 
                               window = torch.hann_window(n_fft).to(curr_proj.device),
                               return_complex = True) # G G+ cn  

        curr_proj_phase = torch.angle(curr_proj) 
        curr_proj = spectrogram * torch.exp(1j * curr_proj_phase)   # Pc1(Pc2(cn-1))  
            
        X = curr_proj + alpha * (curr_proj - prev_proj)
        prev_proj = curr_proj

    x = torch.istft(X, 
                    n_fft = n_fft,
                    window = torch.hann_window(1024).to(X.device))

    return x


def initialize_phase(spectrogram, init = "zeros"):
    
    if init == "zeros":
        X_init_phase = torch.zeros_like(spectrogram)    
    elif init =="random":
        X_init_phase = torch.pi * (2 * torch.rand_like(spectrogram) - 1)
    else:
        raise ValueError(f"init must be 'zeros' or 'random', received: {init}")
    
    return X_init_phase

ao = torch.rand((513, 128), dtype=torch.complex64)
fast_griffin_lim(ao)