import argparse, yaml
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .models.transformer import  FFTransformer
from .models.encoder import Encoder

import torch

NUM_CLASSES = 1000

def create_model(config):
    cEnc, cLat, cDec = config["encoder"], config["latent"], config["decoder"]

    models = {}

    if cLat['latent_size'] != 0:
        models['encoder'] = Encoder(cEnc['dim_internal'],cEnc['n_blocks'],cEnc['n_heads'],out_d=cLat['latent_size'],length=cEnc['len_enc']*32)

    models['eps_model'] = FFTransformer(cDec['dim_internal'],cDec['n_blocks'],cDec['n_heads'],learn_sigma=config['diffusion']['learn_sigma'],d_cond=cLat['latent_size'])

    return torch.nn.ModuleDict(models)

def create_gaussian_diffusion(config):
    cDiff = config['diffusion']
    betas = gd.get_named_beta_schedule(cDiff['noise_schedule'], cDiff['diffusion_steps'])
    if cDiff['use_kl']:
        loss_type = gd.LossType.RESCALED_KL
    elif cDiff['rescale_learned_sigmas']:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    timestep_respacing = cDiff['timestep_respacing']
    if not timestep_respacing:
        timestep_respacing = [cDiff['diffusion_steps']]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(cDiff['diffusion_steps'], timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not cDiff['predict_xstart'] else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cDiff['sigma_small']
                else gd.ModelVarType.FIXED_SMALL
            )
            if not cDiff['learn_sigma']
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=cDiff['rescale_timesteps'],
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def merge_configs(default_config, config):
    for k, v in default_config.items():
        if k not in config:
            config[k] = v
        elif isinstance(v, dict):
            merge_configs(v, config[k])
    return config

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    default_config = yaml.safe_load(open('config/default.yaml', 'r'))
    config = yaml.safe_load(open(args.config, 'r'))

    config = merge_configs(default_config, config)
    return config