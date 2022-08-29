"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from utils.pianoroll import PianoRoll

def main():
    args = create_argparser().parse_args()

    logger.configure(dir=os.path.dirname(args.model_path))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    encoder = model['encoder'] if 'encoder' in model else None
    eps_model = model['eps_model']

    logger.log("sampling...")
    save_path = os.path.join(logger.get_dir(),'samples/',os.path.basename(args.model_path).rsplit( ".", 1 )[ 0 ]+'/')
    all_images = []
    all_labels = []
    i=0
    while len(all_images) < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        shape = (args.batch_size, args.segment_length if args.segment_length !=0 else 180*32, 88)

        if encoder is not None:
            model_kwargs["condition"] = generate_latent(args, encoder, save_path, i)

        # same noise for the whole batch
        # noise = th.randn(1,*shape[1:],device=dist_util.dev()).expand(shape)
        sample = sample_fn(
            eps_model,
            shape,
            clip_denoised=args.clip_denoised,
            #noise = noise,
            model_kwargs=model_kwargs,
        )
        for s in sample:
            # save midi
            path = os.path.join(logger.get_dir(),'samples/',os.path.basename(args.model_path).rsplit( ".", 1 )[ 0 ]+'/', f"{i}.mid")
            i+=1
            os.makedirs(os.path.dirname(path), exist_ok=True)
            PianoRoll.from_tensor((s+1)*64,thres = 20).to_midi(path)

        all_images+=[s for s in sample]
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            #dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) } samples")

    catted = th.cat(all_images,0) # time dim
    PianoRoll.from_tensor((catted+1)*64,thres = 5).to_midi(save_path+"all.mid")
    logger.log("sampling complete")

def generate_latent(args, encoder, save_path, suffix = ''):
    if args.latent_mode == 'all_random': latent = th.randn(args.batch_size, 16).to(dist_util.dev())
    if args.latent_mode == 'same': latent = th.randn(1, 16).to(dist_util.dev()).expand(args.batch_size, -1)
    if args.latent_mode == 'interpolate':
        if a is None or b is None: # generate a and b randomly
            a = th.randn(16).to(dist_util.dev())
            b = -a
        else: # use a and b from args
            a = PianoRoll.load(args.a).get_random_tensor_clip(args.segment_length,normalized=True).to(dist_util.dev())
            b = PianoRoll.load(args.b).get_random_tensor_clip(args.segment_length,normalized=True).to(dist_util.dev())
            PianoRoll.from_tensor(a,thres = 0,normalized=True).to_midi(save_path+f'a{suffix}.mid')
            PianoRoll.from_tensor(b,thres = 0,normalized=True).to_midi(save_path+f'b{suffix}.mid')
            with th.no_grad():
                a = encoder(a.unsqueeze(0)).squeeze(0)
                b = encoder(b.unsqueeze(0)).squeeze(0)
        latent = th.stack([(a*(1-j)+b*j)/(args.batch_size-1) for j in range(args.batch_size)])
    if args.latent_mode == 'reconstruct': 
        a = PianoRoll.load(args.a).get_random_tensor_clip(args.segment_length,normalized=True).to(dist_util.dev())
        PianoRoll.from_tensor(a,thres = 0,normalized=True).to_midi(save_path+f'a{suffix}.mid')
        with th.no_grad():
            a = encoder(a.unsqueeze(0)).squeeze(0)
        latent = a.expand(args.batch_size, -1)
    return latent

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=16,
        use_ddim=False,
        model_path="",
        segment_length = 0,
        latent_mode = 'all_random',
        a = None, # for interpolate latent mode
        b = None, # for interpolate latent mode
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
