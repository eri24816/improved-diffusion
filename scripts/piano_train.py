"""
Train a diffusion model on images.
"""

import yaml, os

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import load_data
from improved_diffusion.script_util import get_config, create_model, create_gaussian_diffusion
from improved_diffusion.train_util import TrainLoop


def main():
    config, config_override, conf_path = get_config(return_path=True) # read config yaml at --config
    
    dist_util.setup_dist()
    name = config['name'] if config['name'] != '' else os.path.basename(conf_path).rsplit( ".", 1 )[ 0 ]
    log_dir = os.path.join('./log',name)
    logger.configure(tb=True, dir=log_dir)

    logger.set_level({"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "DISABLE": 50}[config["log_level"]])

    logger.log("")
    logger.log('-'*30)
    logger.log('Config:')
    logger.log(yaml.dump(config_override))
    logger.log('-'*30)
    logger.log("")

    logger.log("creating model and diffusion...")
    model = create_model(config)
    model.to(dist_util.dev())
    diffusion = create_gaussian_diffusion(config)
    
    logger.log("creating data loader...")
    data = load_data(
        data_dir=config['data_dir'],
        batch_size=config['training']['global_batch_size']//dist_util.get_world_size(),
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
