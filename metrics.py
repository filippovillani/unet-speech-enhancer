import tensorflow as tf
import numpy as np
from pypesq import pesq

from utils import waveform_from_spectrogram

# Scale Invariant Noise to Signal Ratio loss function
class SI_NSR_loss(tf.keras.losses.Loss):
    def __init__(self, name='SI-NSR'):
        super().__init__(name='SI-NSR')

    def call(self, clean_speech, enhanced_speech):
        s_target = tf.math.divide(tf.math.multiply(tf.math.reduce_sum(tf.math.multiply(clean_speech, enhanced_speech)), clean_speech),
                                  tf.math.reduce_sum(tf.math.pow(clean_speech, 2)))
        
        e_noise = enhanced_speech - s_target
        SI_NSR_linear = tf.math.divide(tf.math.reduce_sum(tf.math.pow(e_noise, 2)), tf.math.reduce_sum(tf.math.pow(s_target, 2)))
        # tf doesn't provide a log10 function
        SI_NSR = tf.math.multiply(tf.math.divide(tf.math.log(SI_NSR_linear), tf.math.log(10.)), 10.)
        
        return SI_NSR

# SI-SNR metric
def create_snr_metric():
    def snr_metric(clean_speech, enhanced_speech):
        s_target = tf.math.divide(tf.math.multiply(tf.math.reduce_sum(tf.math.multiply(clean_speech, enhanced_speech)), clean_speech),
                                  tf.math.reduce_sum(tf.math.pow(clean_speech, 2)))
        
        e_noise = enhanced_speech - s_target
        SI_SNR_linear = tf.math.divide(tf.math.reduce_sum(tf.math.pow(s_target, 2)), tf.math.reduce_sum(tf.math.pow(e_noise, 2)))

        return tf.math.multiply(tf.math.divide(tf.math.log(SI_SNR_linear), tf.math.log(10.)), 10.)
    return snr_metric

class SI_SNR(tf.keras.metrics.Metric):
    def __init__(self, name='si_snr_metric'):
        super(SI_SNR, self).__init__(name=name)
        self.snr_fn = create_snr_metric()
        self.snr_metric = self.add_weight(name='si_snr', initializer='zeros')
    
    def update_state(self, clean_speech, enhanced_speech, sample_weight=None):
        self.snr_metric.assign(self.snr_fn(clean_speech, enhanced_speech))
   
    def result(self):
        return self.snr_metric.value()        


def create_pesq_metric(batch_size=16, sr=16000):
    def pesq_metric(clean_speech, enhanced_speech):
        pesq_batch = []
        # For time efficiency I'm using just half of the batch
        for el in range(int(batch_size/2)):
            # squeeze channel and batch dimensions
            clean_speech_element = tf.squeeze(clean_speech[el])
            enhanced_speech_element = tf.squeeze(enhanced_speech[el])
            # compute waveforms from spectrograms (n_iter is very low, this will affect the PESQ score)
            clean_speech_element = waveform_from_spectrogram(clean_speech_element, n_iter=128)           
            enhanced_speech_element = waveform_from_spectrogram(enhanced_speech_element, n_iter=128)
            # compute pesq
            pesq_value = pesq(clean_speech_element, enhanced_speech_element, sr)
            pesq_batch.append(pesq_value)
            
        return tf.reduce_mean(pesq_batch)
    return pesq_metric

class PESQ(tf.keras.metrics.Metric):
    def __init__(self, name='pesq_metric', sr=16000):
        super(PESQ, self).__init__(name=name)
        self.sr = sr
        self.pesq_fn = create_pesq_metric(self.sr)
        self.pesq_metric = self.add_weight(name='pesq_weight', initializer='zeros')
    
    def update_state(self, clean_speech, enhanced_speech, sample_weight=None):
        self.pesq_metric.assign(self.pesq_fn(clean_speech, enhanced_speech))
    def result(self):
        return self.pesq_metric.value()

# This is a custom evaluation loop that computes the PESQ metric
# The output will be something like [PESQ_taskA, PESQ_taskB, ...]
pesq_metric = PESQ()

def pesq_eval_model(model, test_sets, batch_for_test):
    num_tasks = len(test_sets)
    metrics = []

    for task in range(num_tasks):
        metric_task = []
        for data in test_sets[task].take(batch_for_test):
            noisy, clean = data[0], data[1]
           # this if statement could be useless, just keep the else part  
            if num_tasks == 1:
                pesq_metric.update_state(clean, model(noisy, training=False))
            # select the right head for the task
            else:
                pesq_metric.update_state(clean, model(noisy, training=False)[task])
            metric_task.append(pesq_metric.result().numpy())
        
        pesq_metric.reset_states()
         
        if batch_for_test > 1:
            metric_task = np.mean(metric_task)

        print(f'PESQ on test set: {(metric_task[0]):.2f}')
        metrics.append(metric_task)

    return metrics