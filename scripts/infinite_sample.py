"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import yaml
import os

import torch as th
import torch.distributed as dist
import random

from improved_diffusion import dist_util, guiders
from improved_diffusion.script_util import get_config, create_model, create_gaussian_diffusion


from utils.pianoroll import PianoRoll
from tqdm import tqdm

def main():
    config, config_override = get_config() # read config yaml at --config
    dist_util.setup_dist()
    
    conf = config['sampling']
    len_enc= config['encoder']['len_enc']
    len_dec = config['decoder']['len_dec']
    
    '''
    setup model
    '''
    print("creating model and diffusion...")
    model = create_model(config)
    diffusion = create_gaussian_diffusion(config)
    model.load_state_dict(dist_util.load_state_dict(conf["model_path"], map_location="cpu"))
    model.to(dist_util.dev()).eval()
    encoder,eps_model = model['encoder'] if 'encoder' in model else None, model['eps_model']
    
    '''
    setup guider
    '''
    start_tick = 0
    base_song = PianoRoll.from_midi("log/input/untitlessadd.mid")
    exp_name = f'inf/{base_song.metadata.name}'
    base_song = base_song.to_tensor(normalized=True,start_time=start_tick,end_time=start_tick+32*len_dec,padding=True)

    '''
    generate samples
    '''
    print("sampling...")
    save_path = os.path.join(os.path.dirname(conf["model_path"]),'samples/',os.path.basename(conf["model_path"]).rsplit( ".", 1 )[ 0 ]+'/',exp_name)
    os.makedirs(save_path, exist_ok=True)
    base_song = th.repeat_interleave(base_song.unsqueeze(0),conf['batch_size'],dim=0)
    all_samples = []
    all_samples.append(base_song[:,:32*16])
    prompt = base_song[:,32*8:32*16]

    sample_fn = diffusion.p_sample_loop if not conf['use_ddim'] else diffusion.ddim_sample_loop

    shape = (conf['batch_size'], len_dec*32 , 88)

    for i in range(32): # tqdm?
        noise = th.randn(shape,device=dist_util.dev())

        padded_prompt = th.cat([prompt,th.zeros_like(prompt)-0.3],1)
        mb = guiders.MaskBuilder(padded_prompt[0])
        guider = guiders.ReconstructGuider(padded_prompt,mb.FirstBars(8),3,None)
        guider.reset(noise)

        sample = sample_fn(eps_model,shape,progress=True,clip_denoised=conf['clip_denoised'],noise = noise,model_kwargs={},guider=guider)
        sample = sample.detach()

        all_samples+=[sample[:,32*8:].cpu()]
        print(len(all_samples))
        prompt = sample[:,32*8:]

        if i%2==0:
            catted = th.cat(all_samples,dim=1)
            for i, pr in enumerate(catted):
                PianoRoll.from_tensor((pr+1)*64,thres = 5).to_midi(os.path.join(save_path, f'{i}.mid'))

    print("sampling complete")


if __name__ == "__main__":
    main()