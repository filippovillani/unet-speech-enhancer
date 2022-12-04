import os
import json
import torch
from time import time
from tqdm import tqdm

from dataset import build_dataloaders
from model import UNet
from metrics import si_nsr_loss, si_snr_metric
import config


def eval_model(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader)->torch.Tensor:

    score = 0.
    loss = 0.
    model.eval()
    
    with torch.no_grad():
        for n, batch in enumerate(tqdm(dataloader)):
            noisy_speech, clean_speech = batch["noisy"], batch["speech"]

            enhanced_speech = model(noisy_speech)
            snr_metric = si_snr_metric(enhanced_speech, clean_speech)
            score += ((1./(n+1))*(snr_metric-score))

            nsr_loss = si_nsr_loss(enhanced_speech, clean_speech)
            loss += ((1./(n+1))*(nsr_loss-loss))
    
    return score, loss

        

def train_model(args, hparams):
    
    training_state_path = config.RESULTS_DIR / (args.experiment_name+"_train_state.json")
    
    model = UNet().double()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=hparams.lr)

    if args.weights_dir is not None:
        weights_dir = config.WEIGHTS_DIR / args.weights_dir
        weights_path = weights_dir / args.weights_dir
        opt_path = weights_dir / (args.weights_dir + '_opt')
        
        model.load_state_dict(torch.load(weights_path))
        optimizer.load_state_dict(torch.load(opt_path))        
        
        with open(training_state_path, "r") as fr:
            training_state = json.load(fr)
         
    else:
        weights_dir = config.WEIGHTS_DIR / args.experiment_name
        weights_path = weights_dir / args.experiment_name
        opt_path = weights_dir / (args.experiment_name + '_opt')
        
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
            
        training_state = {"epochs": 1,
                          "patience_epochs": 0,  
                          "best_val_loss": 9999,
                          "best_val_score": 0,
                          "best_epoch": 0,
                          "train_loss_hist": [],
                          "val_loss_hist": [],
                          "val_score_hist": []}

    # Build training and validation 
    train_dl, val_dl, _ = build_dataloaders(config.DATA_DIR, hparams) 

    print('_____________________________')
    print('       Training start')
    print('_____________________________')
    while training_state["patience_epochs"] < hparams.patience and training_state["epochs"] <= hparams.epochs:
        print(f'\n§ Train epoch: {training_state["epochs"]}\n')
        
        model.train()
        train_loss = 0.
        start_epoch = time()        
   
        for n, batch in enumerate(tqdm(train_dl, desc=f'Epoch {training_state["epochs"]}')):   
            optimizer.zero_grad()  
            noisy_speech, clean_speech = batch["noisy"], batch["speech"]
            enhanced_speech = model(noisy_speech.double())
            loss = si_nsr_loss(enhanced_speech, clean_speech)
            train_loss += ((1./(n+1))*(loss-train_loss))
            loss.backward()  
            optimizer.step()
        
        training_state["train_loss_hist"].append(train_loss.item())
        print(f'\nTraining loss:     {training_state["train_loss_hist"][-1]:.4f}')
        
        # Evaluate on the validation set
        val_score, val_loss = eval_model(model=model, 
                                         dataloader=val_dl)
        
        training_state["val_loss_hist"].append(val_loss.numpy())
        training_state["val_score_hist"].append(val_score.numpy())
        
        print(f'\nValidation Loss:   {val_loss:.4f}\n')
        print(f'\nValidation SI_SNR: {val_score:.4f}\n')
        
        if val_score <= training_state["best_val_score"]:
            training_state["patience_epochs"] += 1
            print(f'Best epoch was Epoch {training_state["best_epoch"]}')
        else:
            training_state["patience_epochs"] = 0
            training_state["best_val_score"] = val_score.numpy()
            training_state["best_val_loss"] = val_loss.numpy()
            training_state["best_epoch"] = training_state["epochs"]
            print("SI-SNR on validation set improved\n")
            # Save the best model
            torch.save(model.state_dict(), weights_path)
            torch.save(optimizer.state_dict(), opt_path)
                    
        with open(training_state_path, "w") as fw:
            json.dump(training_state, fw)

        training_state["epochs"] += 1 

        print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int((((time()-start_epoch))%60)*60/100)} s')
        print('_____________________________')

    print('Best epoch on Epoch ', training_state["best_epoch"])    
    print('val SI-NSR Loss:  \t', training_state["val_loss_hist"][training_state["best_epoch"]-1])
    print('val SI-SNR Score: \t', training_state["best_val_score"])
    print('____________________________________________')