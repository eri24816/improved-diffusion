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
from utils import pianoroll

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

    logger.log("sampling...")
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
        # same noise for the whole batch
        # noise = th.randn(1,*shape[1:],device=dist_util.dev()).expand(shape)
        sample = sample_fn(
            model,
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
        
        '''
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.unsqueeze(1)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        '''

        all_images+=[s for s in sample]
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            #dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) } samples")

    '''
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{os.path.basename(args.model_path).split('.')[0]}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    '''
    catted = th.cat(all_images,0) # time dim
    path = os.path.join(logger.get_dir(),'samples/',os.path.basename(args.model_path).rsplit( ".", 1 )[ 0 ]+'/', "all.mid")
    PianoRoll.from_tensor((catted+1)*64,thres = 5).to_midi(path)
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=16,
        use_ddim=False,
        model_path="",
        segment_length = 0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
