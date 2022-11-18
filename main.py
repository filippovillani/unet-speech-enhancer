import argparse

import tensorflow as tf

from datasets import build_datasets
from model import UNet, unet
from metrics import SI_NSR_loss, SI_SNR
from training import train_model
import config

# train_config = {'batch_size': 16,
#                 'epochs': 30,
#                 'patience': 5,
#                 'lr': 1e-3}
# # class Config:
# #     def __init__(self):
# #         self.batch_size = 16
# #         self.epochs = 30
# #         self.patience = 5
# #         self.lr = 1e-3

# # train_config = Config()
# n_mels = 96

# # main_dir = Path(__file__).parent
# # data_dir = main_dir / "data"

# train_ds, val_ds, test_ds = build_datasets(config.DATA_DIR, train_config.batch_size)


# loss_fn = SI_NSR_loss()
# snr_metric = SI_SNR()

# enhancer = UNet()
# train_model(train_config, config.DATA_DIR)
# # enhancer = unet((n_mels, 248, 1))
# opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

# enhancer.compile(optimizer=opt, loss=loss_fn, metrics=[snr_metric])

# callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
#                   tf.keras.callbacks.ModelCheckpoint('unet_reduced_pesq01.hdf5', save_best_only=True)]

# entire_dataset_history = enhancer.fit(train_ds, 
#                                       validation_data=val_ds, 
#                                       epochs=train_config.epochs, 
#                                       callbacks=callbacks_list)


#%%
def parse_args():
    """
    Returns
    -------
    args : 
        - train, predict
        - experiment_name, opt
        - model_weights_path, opt
        - batch_size, def 16
        - epochs    , def 30
        - patience  , def 10
        - lr        , def 1e-3

    """
    
    parser = argparse.ArgumentParser(description='U-Net Speech Enhancer',
                                     epilog='I hope it sounds good')
    subparsers = parser.add_subparsers(help='help for subcommand', dest="subcommand")
    
    # Train commands
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('experiment_name', 
                              type=str, 
                              help='Choose a name for your experiment', 
                              default='unet0')
    parser_train.add_argument('--weights_path', 
                              type=str, 
                              help='If you want to restart the training, specify the weigths location')
    parser_train.add_argument('--batch_size', 
                              type=int,
                              help='Batch size for training',
                              default=16)
    parser_train.add_argument('--epochs', 
                              type=int,
                              help='Number of epochs to train the model',
                              default=30)
    parser_train.add_argument('--patience', 
                              type=int,
                              help='Patience parameter for early-stopping',
                              default=10)
    parser_train.add_argument('--lr', 
                              type=float,
                              help='Learning Rate for training',
                              default=1e-3)
    # Predict commands
    parser_b = subparsers.add_parser('predict', help='Use the model for prediction')
    parser_b.add_argument('--weights_path', type=str, help='help for b')
    
    # parser.add_argument('-a', 
    #                     type=int,
    #                     default=3)
    # parser.add_argument('-b', 
    #                     type=int,
    #                     default=5,
    #                     choices=[4, 5])
    
    args = parser.parse_args()
    return args

    
def main(args):
    train_model(args, config.DATA_DIR)
    # pass


if __name__ == "__main__":
    args = parse_args()   
    main(args)