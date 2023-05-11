import argparse

import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from dataset import build_dataloaders
from metrics import SI_SSDR
from networks.UNet.models import UNet
from networks.PInvDAE.models import PInvDAE
from networks.DeGLI.models import DeGLI
from utils.audioutils import denormalize_db_spectr, to_linear, normalize_db_spectr, to_db, compute_wav
from utils.utils import load_config, save_json


class Tester:
    def __init__(self, args):
        
        super(Tester, self).__init__()
        self.experiment_name = args.experiment_name
        self.melspec2spec_name = args.melspec2spec
        self.degli_name = args.degli
        self.model = args.model
        self._set_paths()
        
        self.hprms = load_config(self.config_path)
        self.hprms.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hprms.batch_size = 1
        
        # Metrics
        self.metrics = self._metrics_to_monitor(args.model)
        self.sissdr = SI_SSDR()
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hprms.sr, mode="wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hprms.sr)
        
        # Models
        self.enh_model = UNet(self.hprms).to(self.hprms.device)
        self.enh_model.load_state_dict(torch.load(self.enh_weights_path))  
        self.enh_model.eval()
        
        if self.model in ['melspec2spec', 'degli']:
            self.melspec2spec_hprms = load_config(self.melspec2spec_config_path)
            self.melspec2spec_hprms.device = self.hprms.device
            self.melspec2spec_model = PInvDAE(self.melspec2spec_hprms).to(self.melspec2spec_hprms.device)
            self.melspec2spec_model.load_state_dict(torch.load(self.melspec2spec_weights_path))
            self.melspec2spec_model.eval()
            
            if self.model == 'degli':
                self.degli_hprms = load_config(self.degli_config_path)
                self.degli_hprms.device = self.hprms.device
                self.degli_model = DeGLI(self.degli_hprms).to(self.degli_hprms.device)
                self.degli_model.load_state_dict(torch.load(self.degli_weights_path))
                self.degli_model.eval()
    
    def _metrics_to_monitor(self, model):
        
        if model in ['enhancer', 'enhancer_hz']:
            metrics = ['si-ssdr-enh']
        elif model == 'melspec2spec':
            metrics = ['si-ssdr-pinv', 'stoi-noisy', 'pesq-noisy']    
        elif model == 'degli':
            metrics = ['stoi-degli', 'pesq-degli']    
        
        return metrics
            
    def _set_paths(self):
        
        enh_dir = config.MODELS_DIR / self.experiment_name
        
        self.enh_weights_path = enh_dir / 'weights' / 'best_weights'
        self.metrics_path = enh_dir / 'test_metrics.json'
        self.config_path = enh_dir / 'config.json'
        
        if self.model in ['melspec2spec', 'degli']:
            melspec2spec_dir = config.MODELS_DIR / self.melspec2spec_name
            self.melspec2spec_weights_path = melspec2spec_dir / 'weights' / 'best_weights'   
            self.melspec2spec_config_path = melspec2spec_dir / 'config.json'
            if self.model == 'degli':
                degli_dir = config.MODELS_DIR / self.degli_name
                self.degli_weights_path = degli_dir / 'weights' / 'best_weights'   
                self.degli_config_path = degli_dir / 'config.json'
        
    def evaluate(self, test_dl):
  
        test_scores = {"si-ssdr-enh": 0.,
                       "si-ssdr-pinv": 0.,
                       "stoi-noisy": 0.,
                       "pesq-noisy": 0.,
                       "stoi-degli": 0.,
                       "pesq-degli": 0.}      
          
        pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
        with torch.no_grad():
            for n, batch in enumerate(pbar):   
                
                noisy_phasegram = torch.angle(batch["noisy_stft"]).float().to(self.hprms.device)
                speech_wav = batch["speech_wav"].float().to(self.hprms.device).squeeze()
                speech_stft = batch["speech_stft"].to(self.hprms.device)
                speech_spec = torch.abs(speech_stft).float()
                
                if self.model == 'enhancer_hz':
                    # Compute output
                    noisy_speech = normalize_db_spectr(to_db(torch.abs(batch["noisy_stft"]))).float().unsqueeze(1).to(self.hprms.device) 
                    enhanced_speech = self.enh_model(noisy_speech).squeeze(1)
                    
                    # Metrics
                    sissdr_score = self.sissdr(enhanced_speech, speech_spec).cpu().item()
                    test_scores["si-ssdr-enh"] += ((1./(n+1))*(sissdr_score-test_scores["si-ssdr-enh"]))

                else:
                    # Compute Mel-Enhancer's output
                    speech_mel_db = batch["speech_mel_db_norm"].float().to(self.hprms.device)
                    noisy_speech = batch["noisy_mel_db_norm"].float().to(self.hprms.device) 
                    enhanced_speech = self.enh_model(noisy_speech)  # ENHANCER              
                        
                    # Metrics
                    sissdr_score = self.sissdr(to_linear(denormalize_db_spectr(enhanced_speech)), 
                                               to_linear(denormalize_db_spectr(speech_mel_db))).cpu().item()
                    test_scores["si-ssdr-enh"] += ((1./(n+1))*(sissdr_score-test_scores["si-ssdr-enh"]))    
                        
                    if self.model in ['melspec2spec', 'degli']:
                        # Compute melspec2spec's output
                        enhanced_speech = self.melspec2spec_model(enhanced_speech).squeeze(1)  # MELSPEC2SPEC
                        enhanced_speech = to_linear(denormalize_db_spectr(enhanced_speech)).squeeze(1) 
                        sissdr_score = self.sissdr(enhanced_speech, speech_spec).cpu().item()
                        test_scores["si-ssdr-pinv"] += ((1./(n+1))*(sissdr_score-test_scores["si-ssdr-pinv"]))
                        
                        # METRICS
                        enhanced_speech_stft = (enhanced_speech * torch.exp(1j * noisy_phasegram)).squeeze(1) 
                        enhanced_wav = compute_wav(enhanced_speech_stft, self.hprms.n_fft)
                        
                        pesq_out = self.pesq(enhanced_wav, speech_wav[:len(enhanced_wav)]).item()
                        test_scores["pesq-noisy"] += ((1./(n+1))*(pesq_out-test_scores["pesq-noisy"]))  
                        stoi_out = self.stoi(enhanced_wav, speech_wav[:len(enhanced_wav)]).item()
                        test_scores["stoi-noisy"] += ((1./(n+1))*(stoi_out-test_scores["stoi-noisy"])) 
                        
                        if self.model == 'degli':
                            # Compute phase reconstruction
                            self.degli_model.repetitions = self.degli_hprms.test_degli_blocks
                            enhanced_speech = self.degli_model(enhanced_speech_stft, enhanced_speech).squeeze(1)   # PHASE
                            # METRICS
                            enhanced_wav = compute_wav(enhanced_speech, self.hprms.n_fft)
                            pesq_out = self.pesq(enhanced_wav, speech_wav[:len(enhanced_wav)]).item()
                            test_scores["pesq-degli"] += ((1./(n+1))*(pesq_out-test_scores["pesq-degli"]))  
                            stoi_out = self.stoi(enhanced_wav, speech_wav[:len(enhanced_wav)]).item()
                            test_scores["stoi-degli"] += ((1./(n+1))*(stoi_out-test_scores["stoi-degli"])) 
                             
                scores_to_print = str({k: round(float(v), 4) for k, v in test_scores.items() if k in self.metrics})
                pbar.set_postfix_str(scores_to_print)
        
        for k,v in test_scores.items():
            print(f'{k} = {v}')
        save_json(test_scores, self.metrics_path)
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment_name', 
                        type=str, 
                        default='enhancer80_02') 
    parser.add_argument('--melspec2spec', 
                        type=str, 
                        default='pinvdae80_02') 
    parser.add_argument('--degli', 
                        type=str, 
                        default='degli80_02') 
    parser.add_argument('--model', 
                        type=str,
                        choices=['enhancer', 'enhancer_hz', 'melspec2spec', 'degli'],
                        help="The model you want to evaluate", 
                        default='degli') 
    args = parser.parse_args()
    
    tester = Tester(args)
    _, _,  test_dl = build_dataloaders(tester.hprms, config.DATA_DIR)
    tester.evaluate(test_dl)