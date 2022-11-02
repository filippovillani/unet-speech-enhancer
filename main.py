import tensorflow as tf
import os

from datasets import build_noisy_speech_df, build_ds_from_df
from model import unet
from metrics import SI_NSR_loss, SI_SNR

batch_size = 16
n_mels = 96

main_dir = os.path.realpath(os.path.join(os.path.dirname('__file__')))
data_main_dir = os.path.join(main_dir, 'data')

noisy_speech_df = build_noisy_speech_df(data_main_dir)
train_ds, val_ds, test_ds = build_ds_from_df(noisy_speech_df, batch_size=batch_size)

for el in test_ds:
    noisy_example, clean_example = tf.squeeze(el[0][0]).numpy(), tf.squeeze(el[1][0]).numpy()
    print('noisy: ', noisy_example.shape, noisy_example.dtype)
    print('clean: ', clean_example.shape, clean_example.dtype)
    break

loss_fn = SI_NSR_loss()
snr_metric = SI_SNR()
#%%
enhancer = unet((n_mels, 248, 1))
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

enhancer.compile(optimizer=opt, loss=loss_fn, metrics=[snr_metric])

callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                  tf.keras.callbacks.ModelCheckpoint('unet_reduced_pesq01.hdf5', save_best_only=True)]

entire_dataset_history = enhancer.fit(train_ds, 
                                      validation_data=val_ds, 
                                      epochs=30, 
                                      callbacks=callbacks_list)