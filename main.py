from pathlib import Path

import tensorflow as tf

from datasets import build_datasets
from model import UNet, unet
from metrics import SI_NSR_loss, SI_SNR
from training import train_model

train_config = {'batch_size': 16,
                'epochs': 30,
                'patience': 5,
                'lr': 1e-3}
class Config:
    def __init__(self):
        self.batch_size = 16
        self.epochs = 30
        self.patience = 5
        self.lr = 1e-3

train_config = Config()
n_mels = 96

main_dir = Path(__file__).parent
data_dir = main_dir / "data"

train_ds, val_ds, test_ds = build_datasets(data_dir, train_config.batch_size)


loss_fn = SI_NSR_loss()
snr_metric = SI_SNR()

enhancer = UNet()
train_model(train_config, data_dir)
# enhancer = unet((n_mels, 248, 1))
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

enhancer.compile(optimizer=opt, loss=loss_fn, metrics=[snr_metric])

# callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
#                   tf.keras.callbacks.ModelCheckpoint('unet_reduced_pesq01.hdf5', save_best_only=True)]

# entire_dataset_history = enhancer.fit(train_ds, 
#                                       validation_data=val_ds, 
#                                       epochs=train_config.epochs, 
#                                       callbacks=callbacks_list)