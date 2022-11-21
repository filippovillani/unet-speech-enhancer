import os
import json

import tensorflow as tf
import numpy as np
from time import time
from tqdm import tqdm

from datasets import build_datasets
from model import UNet
from metrics import SI_NSR_loss, SI_SNR
import config


def eval_model(model, 
               test_set,
               metric,
               loss_fn):

    batch_score = []
    batch_loss = []
    for batch in test_set:
        noisy_speech, clean_speech = batch[0], batch[1]

        pred_speech = model(noisy_speech, training=False)
        metric.update_state(clean_speech, pred_speech)
        batch_score.append(metric.result().numpy())

        loss_ = loss_fn(clean_speech, pred_speech)
        batch_loss.append(loss_)

    score = np.mean(batch_score)
    loss = np.mean(batch_loss)
    
    return score, loss

        

def train_model(args):
    
    training_state_path = config.RESULTS_DIR / (args.experiment_name+"_train_state.json")
    
    model = UNet()

    if args.weights_dir is not None:
        weights_dir = config.WEIGHTS_DIR / args.weights_dir
        weights_path = weights_dir / args.weights_dir
        model.built = True
        model.load_weights(weights_path)
        
        with open(training_state_path, "r") as fr:
            training_state = json.load(fr)
         
    else:
        weights_dir = config.WEIGHTS_DIR / args.experiment_name
        weights_path = weights_dir / args.experiment_name
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
            
        training_state = {"epochs": 1,
                          "patience_epochs": 0,  
                          "best_val_loss": np.Inf,
                          "best_val_score": 0,
                          "best_epoch": 0,
                          "train_loss_hist": [],
                          "val_loss_hist": [],
                          "val_score_hist": []}
         
    # Initialize optimizer, loss_fn and metric
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = SI_NSR_loss()
    metric = SI_SNR()
    # Build training and validation 
    train_ds, val_ds, _ = build_datasets(args.batch_size)
    

    print('_____________________________')
    print('       Training start')
    print('_____________________________')
    while training_state["patience_epochs"] < args.patience and training_state["epochs"] <= args.epochs:
        print(f'\n§ Train epoch: {training_state["epochs"]}\n')

        start_epoch = time()        
        epoch_loss_hist = []
        
        for n, batch in enumerate(tqdm(train_ds, desc=f'Epoch {training_state["epochs"]}')):     
            noisy_speech, clean_speech = batch[0], batch[1]
            with tf.GradientTape() as tape:
                pred_speech = model(noisy_speech, training=True)
                train_loss = loss_fn(clean_speech, pred_speech)
            grads = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_hist.append(train_loss)     
        
        training_state["train_loss_hist"].append(np.mean(epoch_loss_hist))
        print(f'\nTraining loss:     {training_state["train_loss_hist"][-1]:.4f}')
        
        # Evaluate on the valelopment test
        val_score, val_loss = eval_model(model=model, 
                                         test_set=val_ds,
                                         metric=metric,
                                         loss_fn=loss_fn)
        
        training_state["val_loss_hist"].append(val_loss)
        training_state["val_score_hist"].append(val_score)
        
        print(f'\nValidation Loss:   {val_loss:.4f}\n')
        print(f'\nValidation SI_SNR: {val_score:.4f}\n')
        
        if val_score <= training_state["best_val_score"]:
            training_state["patience_epochs"] += 1
            print(f'Best epoch was Epoch {training_state["best_epoch"]}')
        else:
            training_state["patience_epochs"] = 0
            training_state["best_val_score"] = val_score
            training_state["best_val_loss"] = val_loss
            training_state["best_epoch"] = training_state["epochs"]
            print("SI-SNR on validation set improved\n")
            # Save the best model
            model.save_weights(weights_path)
            
        training_state["epochs"] += 1 
        
        with open(training_state_path, "w") as fw:
            json.dump(training_state, fw)

        print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int((((time()-start_epoch))%60)*60/100)} s')
        print('_____________________________')

    print('Best epoch on Epoch ', training_state["best_epoch"])    
    print('val SI-NSR Loss:  \t', training_state["val_loss_hist"][training_state["best_epoch"]-1])
    print('val SI-SNR Score: \t', training_state["best_val_score"])
    print('____________________________________________')
    

