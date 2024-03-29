import argparse
import os
from time import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from dataset import build_dataloaders
from metrics import SI_SSDR
from networks.UNet.models import UNet
from networks.PInvDAE.models import PInvDAE
from networks.DeGLI.models import DeGLI
from utils.plots import plot_train_hist
from utils.audioutils import (denormalize_db_spectr, to_linear, to_db, normalize_db_spectr, create_awgn, standardization)
from utils.utils import (load_config, load_json, save_config, save_json)


class Trainer:
    def __init__(self, args):
        
        super(Trainer, self).__init__()
        self.experiment_name = args.experiment_name
        self.model_trained = args.model
        self.enh_weights_dir = args.enhancer
        self.melspec2spec_weights_dir = args.melspec2spec
        
        self._make_dirs(args.resume_training,
                        args.overwrite)
        self._set_paths()
        self.hprms = self._get_hparams(args.resume_training)
        self.loss_fn = self._get_loss_fn(self.hprms.loss)
        
        
        if self.model_trained == 'melspec2spec':
            self.model2train = PInvDAE(self.hprms).to(self.hprms.device)
            self.enh_model = UNet(self.hprms).to(self.hprms.device)
            self.enh_model.load_state_dict(torch.load(self.enh_weights_path))
        
        elif self.model_trained == 'degli':
            self.model2train = DeGLI(self.hprms).to(self.hprms.device)
            self.melspec2spec_model = PInvDAE(self.hprms).to(self.hprms.device)
            self.melspec2spec_model.load_state_dict(torch.load(self.melspec2spec_weights_path))
            self.enh_model = UNet(self.hprms).to(self.hprms.device)
            self.enh_model.load_state_dict(torch.load(self.enh_weights_path))
            
        elif self.model_trained in ['enhancer', 'enhancer_hz']:
            self.model2train = UNet(self.hprms).to(self.hprms.device)
        
        
        self.optimizer = torch.optim.Adam(params=self.model2train.parameters(),
                                          lr=self.hprms.lr)
        self.lr_sched = ReduceLROnPlateau(self.optimizer, 
                                          factor=0.5, 
                                          patience=self.hprms.lr_patience)
        
        if args.resume_training:
            self.model2train.load_state_dict(torch.load(self.ckpt_weights_path))
            self.optimizer.load_state_dict(torch.load(self.ckpt_opt_path))
            self.lr_sched.load_state_dict(torch.load(self.ckpt_sched_path))
            self.training_state = load_json(self.training_state_path) 
        else:
            self.training_state = {"epochs": 0,
                                    "patience_epochs": 0,  
                                    "best_epoch": 0,
                                    "best_epoch_score": -999,
                                    "train_hist": {},
                                    "val_hist": {}}
            
        self.sissdr = SI_SSDR()
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hprms.sr, mode="wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hprms.sr)    
    
    def train(self, train_dl, val_dl):
        
        print('_____________________________')
        print('       Training start')
        print('_____________________________')
        
        while self.training_state["patience_epochs"] < self.hprms.patience and self.training_state["epochs"] < self.hprms.epochs:
            
            self.training_state["epochs"] += 1 
            print(f'\n§ Train Epoch: {self.training_state["epochs"]}\n')
            
            self.model2train.train()
            train_scores = {"loss": 0.,
                            "si-ssdr": 0.,
                            "pesq": 0.,
                            "stoi": 0.}
            start_epoch = time()        
            pbar = tqdm(train_dl, desc=f'Epoch {self.training_state["epochs"]}', postfix='[]')
        
            for n, batch in enumerate(pbar):   
                    
                self.optimizer.zero_grad()  
                
                # Compute outputs
                if self.model_trained == 'enhancer':
                    speech = batch["speech_mel_db_norm"].float().to(self.hprms.device) # MEL SPEC
                    noisy_speech = batch["noisy_mel_db_norm"].float().to(self.hprms.device) 
                    enhanced_speech = self.model2train(noisy_speech)
                        
                elif self.model_trained == 'enhancer_hz':
                    speech = normalize_db_spectr(to_db(torch.abs(batch["speech_stft"]).float().to(self.hprms.device)))
                    noisy_speech = normalize_db_spectr(to_db(torch.abs(batch["noisy_stft"]))).float().unsqueeze(1).to(self.hprms.device) 
                    enhanced_speech = self.model2train(noisy_speech).squeeze(1)
                    
                elif self.model_trained == 'melspec2spec':
                    speech = torch.abs(batch["speech_stft"]).float().to(self.hprms.device)   # STFT SPEC
                    noisy_speech = batch["noisy_mel_db_norm"].float().to(self.hprms.device) 
                    speech = normalize_db_spectr(to_db(speech))
                    with torch.no_grad():
                        enhanced_speech = self.enh_model(noisy_speech)
                    enhanced_speech = self.model2train(enhanced_speech).squeeze(1) # VOCODER

                elif self.model_trained == 'degli':
                    self.model2train.repetitions = 1
                    speech = batch["speech_stft"].to(self.hprms.device)
                    noisy_speech = batch["noisy_mel_db_norm"].float().to(self.hprms.device) 
                    with torch.no_grad():
                        enhanced_speech = self.enh_model(noisy_speech)
                        enhanced_speech = self.melspec2spec_model(enhanced_speech) 
                        
                        enhanced_speech = to_linear(denormalize_db_spectr(enhanced_speech))
                        enhanced_speech = (enhanced_speech * torch.exp(1j * torch.angle(speech))).squeeze(1)
                        noise = create_awgn(enhanced_speech, 
                                            max_snr_db = self.hprms.max_awgn_db,
                                            min_snr_db = self.hprms.min_awgn_db)
                        enhanced_awgn_speech = (enhanced_speech + noise).squeeze(1)
                        
                    enhanced_speech = self.model2train(enhanced_awgn_speech, torch.abs(enhanced_speech)).squeeze(1)

                # LOSS and METRICS   
                loss = self.loss_fn(enhanced_speech, speech)
                if self.hprms.weights_decay is not None:
                    l2_reg = self.l2_regularization(self.model2train)
                    loss += self.hprms.weights_decay * l2_reg   
                    
                if (not torch.isnan(loss) and not torch.isinf(loss)): 
                    train_scores["loss"] += ((1./(n+1))*(loss-train_scores["loss"]))
    
                loss.backward()  
                self.optimizer.step() 
                
                sdr_metric = self.sissdr(to_linear(denormalize_db_spectr(enhanced_speech)),
                                         to_linear(denormalize_db_spectr(speech))).detach()
                
                if (not torch.isnan(sdr_metric) and not torch.isinf(sdr_metric)):
                    train_scores["si-ssdr"] += ((1./(n+1))*(sdr_metric-train_scores["si-ssdr"]))
                
                if self.model_trained == 'degli':
                    speech_wav = batch['speech_wav'].float().squeeze().to(self.hprms.device)
                    enhanced_wav = torch.istft(enhanced_speech,
                                               n_fft = self.hprms.n_fft,
                                               window = torch.hann_window(self.hprms.n_fft).to(enhanced_speech.device)).squeeze()
                    enhanced_wav = standardization(enhanced_wav)
                    pesq_metric = self.pesq(enhanced_wav, speech_wav[:len(enhanced_wav)])
                    train_scores["pesq"] += ((1./(n+1))*(pesq_metric-train_scores["pesq"]))
                    stoi_metric = self.stoi(enhanced_wav, speech_wav[:len(enhanced_wav)])
                    train_scores["stoi"] += ((1./(n+1))*(stoi_metric-train_scores["stoi"]))
                   
                scores_to_print = str({k: round(float(v), 4) for k, v in train_scores.items() if v!= 0})
                pbar.set_postfix_str(scores_to_print)

                # if n == 5:
                #     break  
                 
            val_scores = self.eval_model(val_dl)
            
            self.lr_sched.step(val_scores["si-ssdr"])
            self._update_training_state(train_scores, val_scores)
            self._save_training_state()
            plot_train_hist(self.training_state_path)
            
            print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int(((time()-start_epoch))%60)} s')
            print('_____________________________')

        print("____________________________________________")
        print("          Training completed")    
        print("____________________________________________")

        return self.training_state
             
             
             
    def eval_model(self,
                   test_dl: DataLoader)->dict:

        self.model2train.eval()

        test_scores = {"loss": 0.,
                        "si-ssdr": 0.,
                        "pesq": 0.,
                        "stoi": 0.}
                
        pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
        with torch.no_grad():
            for n, batch in enumerate(pbar):   
                noisy_speech = batch["noisy_mel_db_norm"].float().to(self.hprms.device) 
                
                if self.model_trained == 'enhancer':
                    speech = batch["speech_mel_db_norm"].float().to(self.hprms.device) # MEL SPEC
                    enhanced_speech = self.model2train(noisy_speech) # ENHANCER
                
                elif self.model_trained == 'enhancer_hz':
                    speech = normalize_db_spectr(to_db(torch.abs(batch["speech_stft"]).float().to(self.hprms.device)))
                    noisy_speech = normalize_db_spectr(to_db(torch.abs(batch["noisy_stft"]))).float().unsqueeze(1).to(self.hprms.device) 
                    enhanced_speech = self.model2train(noisy_speech).squeeze(1)
                        
                elif self.model_trained == 'melspec2spec':
                    speech = torch.abs(batch["speech_stft"]).float().to(self.hprms.device)   # STFT SPEC
                    speech = normalize_db_spectr(to_db(speech))
                    enhanced_speech = self.enh_model(noisy_speech)                 
                    enhanced_speech = self.model2train(enhanced_speech).squeeze(1)  

                elif self.model_trained == 'degli':
                    self.model2train.repetitions = 5
                    speech = batch["speech_stft"].to(self.hprms.device)
                    noisy_phasegram = torch.angle(batch["noisy_stft"]).float().to(self.hprms.device)
                    enhanced_speech = self.enh_model(noisy_speech)                  
                    enhanced_speech = self.melspec2spec_model(enhanced_speech)      
                    enhanced_speech = to_linear(denormalize_db_spectr(enhanced_speech)).squeeze(1) 
                    enhanced_speech_stft = (enhanced_speech * torch.exp(1j * noisy_phasegram)).squeeze(1)                      
                    enhanced_speech = self.model2train(enhanced_speech_stft, enhanced_speech).squeeze(1)  
                       
                loss = self.loss_fn(enhanced_speech, speech)
                test_scores["loss"] += ((1./(n+1))*(loss-test_scores["loss"]))

                if self.model_trained != 'degli':
                    sdr_metric = self.sissdr(to_linear(denormalize_db_spectr(enhanced_speech)),
                                            to_linear(denormalize_db_spectr(speech)))
                    test_scores["si-ssdr"] += ((1./(n+1))*(sdr_metric-test_scores["si-ssdr"]))  
                else:
                    speech_wav = batch['speech_wav'].float().squeeze().to(self.hprms.device)
                    enhanced_wav = torch.istft(enhanced_speech,
                                               n_fft = self.hprms.n_fft,
                                               window = torch.hann_window(self.hprms.n_fft).to(enhanced_speech.device)).squeeze()
                    enhanced_wav = standardization(enhanced_wav)
                    pesq_metric = self.pesq(enhanced_wav, speech_wav[:len(enhanced_wav)])
                    test_scores["pesq"] += ((1./(n+1))*(pesq_metric-test_scores["pesq"]))
                    stoi_metric = self.stoi(enhanced_wav, speech_wav[:len(enhanced_wav)])
                    test_scores["stoi"] += ((1./(n+1))*(stoi_metric-test_scores["stoi"])) 
                
                scores_to_print = str({k: round(float(v), 4) for k, v in test_scores.items() if v != 0})
                pbar.set_postfix_str(scores_to_print)
                
                # if n == 20:
                #     break  
                 
        return test_scores  
    
    def l2_regularization(self, model):
    
        l2_reg = 0.
        for param in model.parameters():
            l2_reg += torch.linalg.norm(param)
            
        return l2_reg

    def _make_dirs(self, resume_training, overwrite):
        
        experiment_dir = config.MODELS_DIR / self.experiment_name
        weights_dir = experiment_dir / "weights"
        
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
            os.mkdir(weights_dir)
        else:
            if not resume_training and not overwrite:
                raise UserWarning('To overvwrite the current experiment use the --overwrite flag')
                        
            
                           
    def _set_paths(self):
        
        experiment_dir = config.MODELS_DIR / self.experiment_name
        weights_dir = experiment_dir / "weights"
        
        self.training_state_path = experiment_dir / "train_state.json"
        self.config_path = experiment_dir / 'config.json'
        
        self.best_weights_path = weights_dir / 'best_weights'
        self.ckpt_weights_path = weights_dir / 'ckpt_weights'
        self.ckpt_opt_path = weights_dir / 'ckpt_opt'
        self.ckpt_sched_path = weights_dir / 'ckpt_sched'
        
        if self.model_trained == 'melspec2spec':
            self.enh_weights_path = config.MODELS_DIR / self.enh_weights_dir / 'weights' / 'best_weights'
        elif self.model_trained == 'degli':
            self.enh_weights_path = config.MODELS_DIR / self.enh_weights_dir / 'weights' / 'best_weights'
            self.melspec2spec_weights_path = config.MODELS_DIR / self.melspec2spec_weights_dir / 'weights' / 'best_weights'
        
    def _get_hparams(self, resume_training):
        
        ''' Creates the hyperparameters if it's the first time,
        otherwise loads the hyperparameters from config.json '''
        
        if resume_training:
            hparams = load_config(self.config_path)
        else:
            hparams = config.create_hparams()
            save_config(hparams, self.config_path)
        
        return hparams
            
    def _get_loss_fn(self, loss: str):
   
        if loss == "l1":
            loss_fn = torch.nn.L1Loss()
        elif loss == "mse":
            loss_fn = torch.nn.MSELoss()
        
        return loss_fn
        
    def _update_training_state(self, train_scores, val_scores):

        train_scores = {k: float(v) for k, v in train_scores.items()}
        val_scores = {k: float(v)  for k, v in val_scores.items()}
        metr = "pesq" if self.model_trained == "degli" else "si-ssdr"
        
        for key, value in train_scores.items():
            if key not in self.training_state["train_hist"]:
                self.training_state["train_hist"][key] = []
            self.training_state["train_hist"][key].append(value)
        
        for key, value in val_scores.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key not in self.training_state["val_hist"]:
                self.training_state["val_hist"][key] = []
            self.training_state["val_hist"][key].append(value)
        
        if val_scores[metr] <= self.training_state["best_epoch_score"]:
            self.training_state["patience_epochs"] += 1
            print(f'\nBest epoch was Epoch {self.training_state["best_epoch"]}: Validation metric = {self.training_state["best_epoch_score"]}')
        else:
            self.training_state["patience_epochs"] = 0
            self.training_state["best_epoch"] = self.training_state["epochs"]
            self.training_state["best_epoch_score"] = val_scores[metr]
            print("Metric on validation set improved")
            
    def _save_training_state(self):
        # Save the best model
        if self.training_state["patience_epochs"] == 0:
            torch.save(self.model2train.state_dict(), self.best_weights_path)
                
        # Save checkpoint to resume training
        save_json(self.training_state, self.training_state_path)  
        torch.save(self.model2train.state_dict(), self.ckpt_weights_path)
        torch.save(self.optimizer.state_dict(), self.ckpt_opt_path)
        torch.save(self.lr_sched.state_dict(), self.ckpt_sched_path)
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment_name', 
                        type=str, 
                        help='Choose a name for the experiment',
                        default='degli80_02') 
    
    parser.add_argument('--model', 
                        type=str, 
                        help="The model you want to train",
                        choices=['enhancer', 'enhancer_hz', 'melspec2spec', 'degli'],
                        default='degli') 

    parser.add_argument('--enhancer', 
                        type=str, 
                        help="Name of the directory containing the enhancer's weights and results. \
                            To be used only if training melspec2spec",
                        default='enhancer80_02')

    parser.add_argument('--melspec2spec', 
                        type=str, 
                        help="Name of the directory containing the melspec2spec's weights and results. \
                            To be used only if training degli",
                        default='pinvdae80_02')
    
    parser.add_argument('--resume_training',
                        action='store_true',
                        help="Use this flag if you want to restart training from a checkpoint")
    
    parser.add_argument('--overwrite',
                        action='store_false', # default should be true 
                        help="Use this flag if you want to overwrite an experiment")
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    train_dl, val_dl, _ = build_dataloaders(trainer.hprms, config.DATA_DIR) 
    trainer.train(train_dl, val_dl)