"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import yaml
import os

import torch as th
import torch.distributed as dist
import random

from improved_diffusion import dist_util, logger, guiders
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

    print(eps_model.transformer[0].temporal_attn.fn.relpb.max_distance)
    
    #base_song = PianoRoll.load('/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll/0.json').slice(32,32+32*len_dec)
    base_song = PianoRoll.from_midi('log/16bar_v_scratch_zero_lm/samples/input/5.mid')
    print('base song:',base_song)
    x_a = []
    for i in range(conf['batch_size']):
        start_tick = random.randint(0,max(0,base_song.duration//32-4))*32
        #start_tick = 8*32
        print(start_tick/32)
        x_a.append(base_song.to_tensor(normalized=True,start_time=start_tick,end_time=start_tick+32*len_dec,padding=True).to(dist_util.dev()))
    x_a = th.stack(x_a,dim=0)
    print(dist_util.dev())
    a_mask = x_a*0 # who use zeros_like XD

    
    a_mask[:,0:32*8] = 1
    #a_mask[:,:,41:] = a_mask[:,0:32*2,:] = 1
    #a_mask[:,0:32*4] = a_mask[:,32*-4:] = 1

    x_a *= a_mask #just to be sure the sampling process doesn't peek the groundtruth in case I implemented it wrong.
    #guider = guiders.ExactGuider(x_a,a_mask,10,diffusion.q_posterior_sample_loop)
    guider = guiders.ExactGuider(x_a,a_mask,10,None)
    exp_name = '5'

    logger.log("sampling...")
    save_path = os.path.join(os.path.dirname(conf["model_path"]),'samples/',os.path.basename(conf["model_path"]).rsplit( ".", 1 )[ 0 ]+'/',exp_name)
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
        noise = th.randn(shape,device=dist_util.dev()).expand(shape)
        guider.reset(noise)
        sample = sample_fn(
            eps_model,
            shape,
            clip_denoised=conf['clip_denoised'],
            noise = noise,
            model_kwargs=model_kwargs,
            denoised_fn=guider.guide,
            progress=True,
        )
        for s in sample:
            # save midi
            path = os.path.join(save_path, f"{i}.mid")
            i+=1
            os.makedirs(save_path, exist_ok=True)
            print('saving',path)
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
