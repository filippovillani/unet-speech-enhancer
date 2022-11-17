from pathlib import Path

import tensorflow as tf

from datasets import build_datasets
from model import unet
from metrics import SI_NSR_loss, SI_SNR


batch_size = 16
n_mels = 96
epochs = 30

main_dir = Path(__file__).parent
data_main_dir = main_dir / "data"

train_ds, val_ds, test_ds = build_datasets(data_main_dir, batch_size)


loss_fn = SI_NSR_loss()
snr_metric = SI_SNR()

enhancer = unet((n_mels, 248, 1))
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

enhancer.compile(optimizer=opt, loss=loss_fn, metrics=[snr_metric])

callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                  tf.keras.callbacks.ModelCheckpoint('unet_reduced_pesq01.hdf5', save_best_only=True)]

entire_dataset_history = enhancer.fit(train_ds, 
                                      validation_data=val_ds, 
                                      epochs=epochs, 
                                      callbacks=callbacks_list)