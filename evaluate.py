import json
import torch
from tqdm import tqdm

import config
from metrics import si_nsr_loss, si_snr_metric
from UNet.models import UNet
from dataset import build_dataloaders

def evaluate(args, hparams):
    
    weights_dir = config.WEIGHTS_DIR / args.weights_dir 
    weights_path = weights_dir / args.weights_dir    
    output_path = config.RESULTS_DIR / (args.weights_dir + '_eval.json')

    model = UNet().double().to(config.device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    _, _,  test_dl = build_dataloaders(config.DATA_DIR, hparams)
    
    with torch.no_grad():
        for n, batch in enumerate(tqdm(test_dl)):
            noisy_speech, clean_speech = batch["noisy"].to(config.device), batch["speech"].to(config.device)

            enhanced_speech = model(noisy_speech)
            snr_metric = si_snr_metric(enhanced_speech, clean_speech)
            score += ((1./(n+1))*(snr_metric-score))

            nsr_loss = si_nsr_loss(enhanced_speech, clean_speech)
            loss += ((1./(n+1))*(nsr_loss-loss))
        
    score = {"si-snr": float(loss.item())}
    print(f'\n SI_SNR = {score["si-snr"]} dB')

    with open(output_path, "w") as fp:
        json.dump(score, fp)
        
