import os
import logging

import numpy as np
import torch

from omegaconf import OmegaConf
from learning.models.score_network import ScoreNetMultiPair

import modelopt.torch.quantization as mtq

def get_cfg(model_dir, model_name='model_best.pth'):
    ckpt_dir = f'{model_dir}/{model_name}'

    cfg = OmegaConf.load(f'{model_dir}/config.yml')

    cfg['ckpt_dir'] = ckpt_dir
    cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in cfg:
      cfg['use_normal'] = False
    if 'use_BN' not in cfg:
      cfg['use_BN'] = False
    if 'zfar' not in cfg:
      cfg['zfar'] = np.inf
    if 'c_in' not in cfg:
      cfg['c_in'] = 4
    if 'normalize_xyz' not in cfg:
      cfg['normalize_xyz'] = False
    if 'crop_ratio' not in cfg or cfg['crop_ratio'] is None:
      cfg['crop_ratio'] = 1.2

    print(f"cfg: \n {OmegaConf.to_yaml(cfg)}")
    return cfg, ckpt_dir

def calibrate_loop(model):
    pass
                
def quantize_model(model):
    quant_cfg = mtq.FP8_DEFAULT_CFG
    with torch.no_grad():
        mtq.quantize(model, quant_cfg,
                            forward_loop=calibrate_loop)
        
if __name__ == "__main__":
    import sys

    cfg, ckpt_dir = get_cfg(sys.argv[1])
    model = ScoreNetMultiPair(cfg, c_in=cfg['c_in']).cuda()
    
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    model.load_state_dict(ckpt)
   
    print(f'begin quantize model...') 
    quantize_model(model)
    print(f'end quantize model...') 
    mtq.print_quant_summary(model)
