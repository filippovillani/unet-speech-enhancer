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
from griffinlim import fast_griffin_lim
from utils.audioutils import denormalize_db_spectr, to_linear, min_max_normalization
from utils.utils import load_config, save_json


class Tester:
    def __init__(self, args):
        
        super(Tester, self).__init__()
        self.experiment_name = args.experiment_name
        self._set_paths()
        
        self.hprms = load_config(self.config_path)
        self.hprms.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hprms.batch_size = 1
        
        self._set_loss(self.hprms.loss)
        self.sissdr = SI_SSDR()
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hprms.sr, mode="wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hprms.sr)
        
        self.model = UNet(self.hprms).to(self.hprms.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device('cpu')))

        self.melfb = torch.as_tensor(melfb(sr = self.hprms.sr, 
                                           n_fft = self.hprms.n_fft, 
                                           n_mels = self.hprms.n_mels)).to(self.hprms.device)
    
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
                        "si-ssdr": 0.,
                        "pesq": 0.,
                        "stoi": 0.}      
          
        pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
        with torch.no_grad():
            for n, batch in enumerate(pbar):   
                
                noisy_speech_melspec = batch["noisy"].float().to(self.hprms.device)
                clean_speech_melspec = batch["speech"].float().to(self.hprms.device)
                noisy_phasegram = batch["noisy_phasegram"].float().to(self.hprms.device)
                # noisy_speech_wav = batch["noisy_speech_wav"].float().to(self.hprms.device).squeeze()
                clean_speech_wav = batch["clean_speech_wav"].float().to(self.hprms.device).squeeze()
                
                enh_speech_melspec = self.model(noisy_speech_melspec)
                
                loss = self.loss_fn(enh_speech_melspec, clean_speech_melspec)
                test_scores["loss"] += ((1./(n+1))*(loss-test_scores["loss"]))
                
                sissdr_out = self.sissdr(to_linear(denormalize_db_spectr(enh_speech_melspec)),
                                         to_linear(denormalize_db_spectr(clean_speech_melspec)))
                # sissdr_in = self.sissdr(to_linear(denormalize_db_spectr(noisy_speech_melspec)),
                #                         to_linear(denormalize_db_spectr(clean_speech_melspec)))
                test_scores["si-ssdr"] += ((1./(n+1))*(sissdr_out-test_scores["si-ssdr"]))  
                
                # test_scores["deltaSISSDR"] = sissdr_out - sissdr_in
                
                # compute enh_speech_stftspec and then fgla
                enh_speech_melspec = to_linear(denormalize_db_spectr(min_max_normalization(enh_speech_melspec)))**2
                enh_speech_stftspec = torch.matmul(torch.linalg.pinv(self.melfb), enh_speech_melspec.squeeze()) # TODO: change with PInvDAE
                enh_speech_stftspec = torch.sqrt(torch.clamp(enh_speech_stftspec, min=0))
                enh_stft_hat = enh_speech_stftspec * torch.exp(1j * noisy_phasegram)
                # enh_speech_wav = fast_griffin_lim(enh_stft_hat, n_iter=200)
                enh_speech_wav = torch.istft(enh_stft_hat,
                                             n_fft = self.hprms.n_fft,
                                             window = torch.hann_window(self.hprms.n_fft).to(enh_stft_hat.device)).squeeze()
                
                pesq_out = self.pesq(enh_speech_wav, clean_speech_wav[:len(enh_speech_wav)])
                test_scores["pesq"] += ((1./(n+1))*(pesq_out-test_scores["pesq"]))  
                # pesq_in = self.pesq(noisy_speech_wav, clean_speech_wav)
                # test_scores["deltaPESQ"] = pesq_out - pesq_in
                
                stoi_out = self.stoi(enh_speech_wav, clean_speech_wav[:len(enh_speech_wav)])
                test_scores["stoi"] += ((1./(n+1))*(stoi_out-test_scores["stoi"])) 
                # stoi_in = self.stoi(noisy_speech_wav, clean_speech_wav)
                # test_scores["deltaSTOI"] = stoi_out - stoi_in
                 
                scores_to_print = str({k: round(float(v), 4) for k, v in test_scores.items()})
                pbar.set_postfix_str(scores_to_print)

        save_json(scores_to_print, self.metrics_path)
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment_name', 
                        type=str, 
                        help='Choose a name for your experiment',
                        default='test00') 
    args = parser.parse_args()
    
    tester = Tester(args)
    _, _,  test_dl = build_dataloaders(tester.hprms, config.DATA_DIR)
    tester.evaluate(test_dl)