import os
from argparse import Namespace

import torch
import yaml
from torch import nn


class Config(Namespace):
    @staticmethod
    def from_dict(config_dict):
        config = Config()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config.from_dict(value)
            setattr(config, key, value)
        return config
    
    @staticmethod
    def from_yaml(path):
        with open(path) as file:
            config_dict = yaml.safe_load(file)
        return Config.from_dict(config_dict)


def save_checkpoint(model, optimizer, scaler, lr_scheduler, step, ckp_interval, retention):
    checkpoint = dict(
        model=model.module.state_dict(),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        lr_scheduler=lr_scheduler.state_dict()
    )
    
    directory = './checkpoints'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    name = 'kangpt'
    
    file_to_remove = f'{name}_{step - ckp_interval * retention}.pt'
    path_to_remove = os.path.join(directory, file_to_remove)
    if os.path.exists(path_to_remove):
        os.remove(path_to_remove)
    
    file_name = f'{name}_{step}.pt'
    save_path = os.path.join(directory, file_name)
    torch.save(checkpoint, save_path)


def count_params(model):
    if isinstance(model, nn.DataParallel):
        n_params = sum(p.numel() for p in model.module.parameters())
    elif isinstance(model, nn.Module):
        n_params = sum(p.numel() for p in model.parameters())
        
    print(f'Parameters: {n_params:,}')
    return n_params
