import argparse, yaml
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .models.transformer import  FFTransformer
from .models.encoder import Encoder
import utils.pianoroll as pianoroll
from . import dist_util

import torch

NUM_CLASSES = 1000

def create_model(config):
    cEnc, cLat, cDec = config["encoder"], config["latent"], config["decoder"]

    models = {}

    if cLat['latent_size'] != 0:
        models['encoder'] = Encoder(cEnc['dim_internal'],cEnc['n_blocks'],cEnc['n_heads'],out_d=cLat['latent_size'],length=cEnc['len_enc']*32)

    init_out = (-1 if config['diffusion']['predict_xstart'] else 0) if cDec['zero'] else None # tensor -1 is midi 0
    models['eps_model'] = FFTransformer(cDec['dim_internal'],cDec['len_dec'] if cDec['spec_num_frames']==-1 else cDec['spec_num_frames'],cDec['n_blocks'],cDec['n_heads'],learn_sigma=config['diffusion']['learn_sigma'],d_cond=cLat['latent_size'],frame_size=cDec['frame_size']*32,init_out=init_out,positional_encoding=cDec['positional_encoding'])

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
        use_loss_mask = cDiff['use_loss_mask']
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

def merge_with_command_line_args(config):
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, config)
    args = parser.parse_args()

    config = args_to_dict(args, config.keys())
    return config

def get_config(path = None, return_path = False):
    default_config = yaml.safe_load(open('config/default.yaml', 'r'))

    if path is not None:
        with open(path, 'r') as f:
            config_override = yaml.safe_load(f)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str)
        args = parser.parse_args()

        config_override = yaml.safe_load(open(args.config, 'r'))

    config = merge_configs(default_config, config_override)

    #config = merge_with_command_line_args(config) #TODO add this back in

    if return_path:
        return config, config_override, args.config
    else:
        return config, config_override


def load_data(
    *, data_dir, batch_size, segment_length = 0, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    # if udistributed training, use the distributed data loader

    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if dist_util.USE_DIST:
        from mpi4py import MPI
        dataset = pianoroll.PianoRollDataset(
            data_dir,
            segment_length,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    else:
        dataset = pianoroll.PianoRollDataset(data_dir, segment_length)
    if deterministic:
        loader = pianoroll.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = pianoroll.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader