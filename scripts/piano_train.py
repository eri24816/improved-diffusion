"""
Train a diffusion model on images.
"""

import argparse, yaml

from improved_diffusion import dist_util, logger
import torch.distributed as dist
from utils.pianoroll import load_data
from improved_diffusion.script_util import get_config, create_model, create_gaussian_diffusion
from improved_diffusion.train_util import TrainLoop


def main():
    config = get_config() # read config yaml at --config
    
    dist_util.setup_dist()
    logger.configure(tb=True)

    logger.set_level({"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "DISABLE": 50}[config["log_level"]])

    logger.log("creating model and diffusion...")
    model = create_model(config)
    model.to(dist_util.dev())
    diffusion = create_gaussian_diffusion(config)
    
    logger.log("creating data loader...")
    data = load_data(
        data_dir=config['data_dir'],
        batch_size=config['training']['global_batch_size']//dist.get_world_size(),
        segment_length= config['decoder']['len_dec']*32, # n_bar * 32
    )

    logger.log("training...")
    TrainLoop(
        config = config,
        model=model,
        diffusion=diffusion,
        data=data,
        **config['training']
    ).run_loop()

if __name__ == "__main__":
    main()
