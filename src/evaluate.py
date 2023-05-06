import argparse

import torch
import torch.nn as nn
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from librosa.filters import mel as melfb
from tqdm import tqdm

import config
from dataset import build_dataloaders
from metrics import SI_SSDR
from networks.UNet.models import UNet
from networks.PInvDAE.models import PInvDAE
from griffinlim import fast_griffin_lim
from utils.audioutils import denormalize_db_spectr, to_linear, min_max_normalization
from utils.utils import load_config, save_json


class Tester:
    def __init__(self, args):
        
        super(Tester, self).__init__()
        self.experiment_name = args.experiment_name
        self.pinvdae_name = args.pinv
        self._set_paths()
        
        self.hprms = load_config(self.config_path)
        self.pinvdae_hprms = load_config(self.pinvdae_config_path)
        self.hprms.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pinvdae_hprms.device = self.hprms.device
        self.hprms.batch_size = 1
        
        self._set_loss(self.hprms.loss)
        self.sissdr = SI_SSDR()
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hprms.sr, mode="wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hprms.sr)
        
        self.enh_model = UNet(self.hprms).to(self.hprms.device)
        self.enh_model.load_state_dict(torch.load(self.enh_weights_path))

        # self.melfb = torch.as_tensor(melfb(sr = self.hprms.sr, 
        #                                    n_fft = self.hprms.n_fft, 
        #                                    n_mels = self.hprms.n_mels)).to(self.hprms.device)

        self.pinvdae_model = PInvDAE(self.pinvdae_hprms).to(self.pinvdae_hprms.device)
        self.pinvdae_model.load_state_dict(torch.load(self.pinvdae_weights_path))
        
    def _set_loss(self, loss: str):
   
        if loss == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss == "mse":
            self.loss_fn = nn.MSELoss()    
            
            
    def _set_paths(self):
        
        enh_dir = config.MODELS_DIR / self.experiment_name
        
        self.enh_weights_path = enh_dir / 'weights' / 'best_weights'
        self.metrics_path = enh_dir / 'test_metrics.json'
        self.config_path = enh_dir / 'config.json'
        
        if self.pinvdae_name is not None:
            pinvdae_dir = config.MODELS_DIR / self.pinvdae_name
            self.pinvdae_weights_path = pinvdae_dir / 'weights' / 'best_weights'   
            self.pinvdae_config_path = pinvdae_dir / 'config.json'
        
    def evaluate(self, test_dl):

        
        self.enh_model.eval()

        test_scores = {"loss": 0.,
                        "si-ssdr": 0.,
                        "pesq": 0.,
                        "stoi": 0.}      
          
        pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
        with torch.no_grad():
            for n, batch in enumerate(pbar):   
                
                noisy_speech_melspec = batch["noisy"].float().to(self.hprms.device)
                clean_speech_melspec = batch["speech"].float().to(self.hprms.device)
                noisy_phasegram = batch["noisy_phasegram"].float().to(self.hprms.device)
                clean_speech_wav = batch["clean_speech_wav"].float().to(self.hprms.device).squeeze()
                                
                enh_speech_melspec = self.enh_model(noisy_speech_melspec)
                
                loss = self.loss_fn(enh_speech_melspec, clean_speech_melspec)
                test_scores["loss"] += ((1./(n+1))*(loss-test_scores["loss"]))
                
                sissdr_out = self.sissdr(to_linear(denormalize_db_spectr(enh_speech_melspec)),
                                         to_linear(denormalize_db_spectr(clean_speech_melspec)))

                test_scores["si-ssdr"] += ((1./(n+1))*(sissdr_out-test_scores["si-ssdr"]))  
                
                
                enh_speech_stftspec = self.pinvdae_model(enh_speech_melspec)
                enh_speech_stftspec = to_linear(denormalize_db_spectr(enh_speech_stftspec)).squeeze(0)

                enh_stft_hat = enh_speech_stftspec * torch.exp(1j * noisy_phasegram)
                enh_speech_wav = torch.istft(enh_stft_hat,
                                             n_fft = self.hprms.n_fft,
                                             window = torch.hann_window(self.hprms.n_fft).to(enh_stft_hat.device)).squeeze()
                
                pesq_out = self.pesq(enh_speech_wav, clean_speech_wav[:len(enh_speech_wav)])
                test_scores["pesq"] += ((1./(n+1))*(pesq_out-test_scores["pesq"]))  
                
                stoi_out = self.stoi(enh_speech_wav, clean_speech_wav[:len(enh_speech_wav)])
                test_scores["stoi"] += ((1./(n+1))*(stoi_out-test_scores["stoi"])) 
                 
                scores_to_print = str({k: round(float(v), 4) for k, v in test_scores.items()})
                pbar.set_postfix_str(scores_to_print)

        save_json(test_scores, self.metrics_path)
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment_name', 
                        type=str, 
                        default='test00') 
    parser.add_argument('--pinv', 
                        type=str, 
                        default='pinvDAE80') 
    args = parser.parse_args()
    
    tester = Tester(args)
    _, _,  test_dl = build_dataloaders(tester.hprms, config.DATA_DIR)
    tester.evaluate(test_dl)