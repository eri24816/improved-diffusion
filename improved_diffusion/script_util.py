import argparse, yaml
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .models.transformer_unet import TransformerUnet, PitchAwareTransformerUnet, FFTransformer
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

'''
def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

'''

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