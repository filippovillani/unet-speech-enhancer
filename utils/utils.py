import json 
from argparse import Namespace



def load_json(path):
    
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def save_json(data, paths):
    
    if not isinstance(data, (list, tuple)):
        data = [data]
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for dat, path in zip(data, paths):
        with open(path, "w") as fp:
            json.dump(dat, fp, indent=4)


def load_config(config_path):
    
    hparams = load_json(config_path)
    hparams = Namespace(**hparams)
    
    return hparams

def save_config(hparams, config_path):
        
    with open(config_path, "w") as fp:
        json.dump(vars(hparams), fp, indent=4)


