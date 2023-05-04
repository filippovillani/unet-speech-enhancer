import argparse
import json

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import build_dataloaders
from metrics import SI_SSDR
from UNet.models import UNet
from utils.audioutils import denormalize_db_spectr, to_linear
from utils.utils import load_config, save_json


class Tester:
    def __init__(self, args):
        
        super(Tester, self).__init__()
        self.experiment_name = args.experiment_name
        self.audio_path = args.audio_path
        self._set_paths()
        
        self.hprms = load_config(self.config_path)
        self.hprms.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._set_loss(self.hprms.loss)
        self.sissdr = SI_SSDR()
        
        self.model = UNet(self.hprms).to(self.hprms.device)
        self.model.load_state_dict(torch.load(self.weights_path))
    
    
    def _set_loss(self, loss: str):
   
        if loss == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss == "mse":
            self.loss_fn = nn.MSELoss()    
            
            
    def _set_paths(self):
        
        results_dir = config.RESULTS_DIR / self.experiment_name
        self.weights_path = config.WEIGHTS_DIR / self.experiment_name / 'best_weights'
        self.metrics_path = results_dir / 'test_metrics.json'
        self.config_path = results_dir / 'config.json'
        
        
    def evaluate(self, test_dl):

        
        self.model.eval()

        test_scores = {"loss": 0.,
                        "si-ssdr": 0.}        
        pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
        with torch.no_grad():
            for n, batch in enumerate(pbar):   
                
                noisy_speech, clean_speech = batch["noisy"].float().to(self.hprms.device), batch["speech"].float().to(self.hprms.device)
                enhanced_speech = self.model(noisy_speech)
                
                loss = self.loss_fn(enhanced_speech, clean_speech)
                test_scores["loss"] += ((1./(n+1))*(loss-test_scores["loss"]))
                
                sdr_metric = self.sissdr(to_linear(denormalize_db_spectr(enhanced_speech)),
                                         to_linear(denormalize_db_spectr(clean_speech)))
                test_scores["si-ssdr"] += ((1./(n+1))*(sdr_metric-test_scores["si-ssdr"]))  
                 
                scores_to_print = str({k: round(float(v), 4) for k, v in test_scores.items()})
                pbar.set_postfix_str(scores_to_print)

        save_json(scores_to_print, self.metrics_path)
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment_name', 
                        type=str, 
                        help='Choose a name for your experiment',
                        default='test00') 
    parser.add_argument('--audio_path',
                        type=str,
                        help='Relative path to .wav audio in mixture_example folder',
                        default='noisy0.wav')
    args = parser.parse_args()
    
    tester = Tester(args)
    _, _,  test_dl = build_dataloaders(config.DATA_DIR, tester.hprms)
    tester.evaluate(test_dl)