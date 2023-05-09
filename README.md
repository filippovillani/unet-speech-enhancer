# Speech Enhancement in Urban Environments
Speech Enhancement is the problem of noise suppression in a speech audio file. A system that can handle such a task can be useful in several scenarios where a speaker is recorded in a noisy environment.

Thinking about a reporter working in the middle of a street and his troup trying to record him, I wondered: is it possible to enhance the reporter's voice while suppressing other sound sources in the scene? Indeed it is, thanks to Machine Learning!

To simulate such a scenario, I mixed speech audio from [DARPA TIMIT Acoustic-Phonetic Continuous Speech](https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech) with environmental noises from [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k).

## Generating noisy data
The clean speeches are mixed with the noise keeping the Signal to Noise Ratio in a range of [-5, +5] dB. <br/>
This is the implementation of the whole mixing and enhancement process:

![generating_noisy_data](https://user-images.githubusercontent.com/93431189/199627651-7c40cd3f-a464-49fe-8e13-8bb205eeafa9.png)

## Examples

<img src="https://user-images.githubusercontent.com/93431189/236959494-80961b36-7644-4140-bff1-aa08fd587de9.mp4" width=50% height=50%/>

