"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import yaml
import os

import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import get_config, create_model, create_gaussian_diffusion


from utils.pianoroll import PianoRoll

def main():
    config, config_override = get_config() # read config yaml at --config
    
    conf = config['sampling']
    len_enc= config['encoder']['len_enc']
    len_dec = config['decoder']['len_dec']
    
    dist_util.setup_dist()
    logger.configure(tb=True)

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

    model.load_state_dict(
        dist_util.load_state_dict(conf["model_path"], map_location="cpu")
    )

    model.to(dist_util.dev())
    model.eval()
    encoder = model['encoder'] if 'encoder' in model else None
    eps_model = model['eps_model']
    

    logger.log("sampling...")
    save_path = os.path.join(logger.get_dir(),'samples/',os.path.basename(conf["model_path"]).rsplit( ".", 1 )[ 0 ]+'/')
    all_images = []
    all_labels = []
    i=0
    while len(all_images) < conf['num_samples']:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not conf['use_ddim'] else diffusion.ddim_sample_loop
        )
        shape = (conf['batch_size'], len_dec*32 if len_dec !=0 else 180*32, 88)

        if encoder is not None:
            model_kwargs["condition"] = generate_latent(conf, encoder, len_dec*32, save_path, i)

        # same noise for the whole batch
        # noise = th.randn(1,*shape[1:],device=dist_util.dev()).expand(shape)
        sample = sample_fn(
            eps_model,
            shape,
            clip_denoised=conf['clip_denoised'],
            #noise = noise,
            model_kwargs=model_kwargs,
        )
        for s in sample:
            # save midi
            path = os.path.join(logger.get_dir(),'samples/',os.path.basename(conf["model_path"]).rsplit( ".", 1 )[ 0 ]+'/', f"{i}.mid")
            i+=1
            os.makedirs(os.path.dirname(path), exist_ok=True)
            PianoRoll.from_tensor((s+1)*64,thres = 20).to_midi(path)

        all_images+=[s for s in sample]
        logger.log(f"created {len(all_images) } samples")

    catted = th.cat(all_images,0)
    PianoRoll.from_tensor((catted+1)*64,thres = 5).to_midi(save_path+"all.mid")
    logger.log("sampling complete")

def generate_latent(conf, encoder, length, save_path, suffix = ''):
    '''
    out : []
    '''
    if conf['latent_mode'] == 'all_random': latent = th.randn(conf['batch_size'], 16).to(dist_util.dev())
    if conf['latent_mode'] == 'same': latent = th.randn(1, 16).to(dist_util.dev()).expand(conf['batch_size'], -1)
    if conf['latent_mode'] == 'interpolate':
        if a is None or b is None: # generate a and b randomly
            a = th.randn(16).to(dist_util.dev())
            b = -a
        else: # use a and b from args
            a = PianoRoll.load(conf['a']).get_random_tensor_clip(length,normalized=True).to(dist_util.dev())
            b = PianoRoll.load(conf['b']).get_random_tensor_clip(length,normalized=True).to(dist_util.dev())
            PianoRoll.from_tensor(a,thres = 0,normalized=True).to_midi(save_path+f'a{suffix}.mid')
            PianoRoll.from_tensor(b,thres = 0,normalized=True).to_midi(save_path+f'b{suffix}.mid')
            with th.no_grad():
                a = encoder(a.unsqueeze(0)).squeeze(0)
                b = encoder(b.unsqueeze(0)).squeeze(0)
        latent = th.stack([(a*(1-j)+b*j)/(conf['batch_size']-1) for j in range(conf['batch_size'])])
    if conf['latent_mode'] == 'reconstruct': 
        a = PianoRoll.load(conf['b']).get_random_tensor_clip(length,normalized=True).to(dist_util.dev())
        PianoRoll.from_tensor(a,thres = 0,normalized=True).to_midi(save_path+f'a{suffix}.mid')
        with th.no_grad():
            a = encoder(a.unsqueeze(0)).squeeze(0)
        latent = a.expand(conf['batch_size'], -1)
    return latent

if __name__ == "__main__":
    main()
