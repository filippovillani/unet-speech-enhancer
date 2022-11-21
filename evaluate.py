import numpy as np
import json

import config
from metrics import SI_SNR
from model import unet
from datasets import build_datasets

def evaluate(args):
    weights_dir = config.WEIGHTS_DIR / args.weights_dir / args.weights_dir
    output_path = config.RESULTS_DIR / (args.weights_dir + '_eval.json')

    model = unet()
    metric = SI_SNR()
    model.built = True
    model.load_weights(weights_dir)

    _, _,  test_set = build_datasets(batch_size=8)
    
    batch_score = []
    for batch in test_set:
        noisy_speech, clean_speech = batch[0], batch[1]
        pred_speech = model(noisy_speech, training=False)
        
        metric.update_state(clean_speech, pred_speech)
        batch_score.append(metric.result().numpy())

    score = {"si-snr": np.mean(batch_score)}
    
    with open(output_path, "w") as fw:
        json.dump(str(score), fw)
        
    print(f'\n SI_SNR = {score["si-snr"]} dB')
