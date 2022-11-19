# Installation
Run from the project's root directory
```
pip install virtualenv
py -m venv env_name
env_name\Scripts\activate  
pip install -r requirements.txt
```

# Speech Enhancement
Speech Enhancement is the problem of noise suppression in a speech audio file. 
A system that can tackle such a task can be beneficial in different scenarios where a speaker is recorded in a noisy environment. 

Thinking about a reporter working in the middle of a street and his troup trying to record him, I was wondering: _is it possible to enhance the reporter's voice suppressing other sound sources in the scene?_
Indeed it is, thanks to Machine Learning!

In order to simulate such a scenario, I decided to mix speech audio from [DARPA TIMIT Acoustic-Phonetic Continuous Speech](https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech) with environmental noises from [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k).

## Generating noisy data
The clean speeches are mixed with the noise keeping the Signal to Noise Ratio in a range of [-5, +5] dB. <br/>
This is the implementation of the whole mixing and enhancement process:
![generating_noisy_data](https://user-images.githubusercontent.com/93431189/199627651-7c40cd3f-a464-49fe-8e13-8bb205eeafa9.png)

