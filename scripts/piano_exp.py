"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import itertools
from math import ceil
import shutil
from typing import Iterable, List
from typing import TypedDict
import os

import torch as th
import torch.distributed as dist
import random

from improved_diffusion import dist_util, guiders, gaussian_diffusion
from improved_diffusion.script_util import get_config, create_model, create_gaussian_diffusion

from utils.pianoroll import PianoRoll
from utils.io_util import json_load, json_dump

class Dimension(TypedDict):
    name: str
    value_range: Iterable
    relevant: bool

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

class ParamSpace:
    def __init__(self, dimensions: List[Dimension]):
        self.dimensions = dimensions
    def get_dict(self):
        return {d['name']: d for d in self.dimensions}
    def all_combinations(self):
        return product_dict(**{d['name']: d['value_range'] for d in self.dimensions})

def init_metadata(num_samples,exp_root_dir, exp_name,param_space:ParamSpace):
    exp_dir = os.path.join(exp_root_dir, exp_name)
    dim_dict = param_space.get_dict()
    dim_dict['sample_idx'] = {'name':'sample_idx','value_range':list(range(num_samples)),'relevant':False}
    metadata = {
        "dimensions": dim_dict,
        "entries": {},
    }
    json_dump(metadata, os.path.join(exp_dir, "metadata.json"))

def sample_with_params(num_samples:int, exp_root_dir, exp_name, params, config, model, diffusion:gaussian_diffusion.GaussianDiffusion, guider: guiders.Guider):
    
    conf = config['sampling']
    len_dec = config['decoder']['len_dec']
    
    model.to(dist_util.dev()).eval()
    encoder,eps_model = model['encoder'] if 'encoder' in model else None, model['eps_model']
    guider.set_diffusion(diffusion)

    params = params.copy()

    print("sampling...")
    exp_dir = os.path.join(exp_root_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    sample_idx = 0
    for _ in range(ceil(num_samples/conf['batch_size'])):
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not conf['use_ddim'] else diffusion.ddim_sample_loop
        )
        shape = (conf['batch_size'], len_dec*32 if len_dec !=0 else 180*32, 88)

        if encoder is not None:
            model_kwargs["condition"] = generate_latent(conf, encoder, len_dec*32, exp_dir)

        # same noise for the whole batch
        min_timestep, max_timestep, noise = guider.get_timestep_range_and_noise()
        if noise is None:
            noise = th.randn(shape,device=dist_util.dev()).expand(shape)
        else:
            noise = noise.unsqueeze(0).expand(shape).to(dist_util.dev())
        guider.reset(noise)

        sample = sample_fn(eps_model,shape,progress=True,
            clip_denoised=conf['clip_denoised'],
            noise = noise,
            model_kwargs=model_kwargs,
            guider=guider,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
        )
        
        metadata = json_load(os.path.join(exp_dir, "metadata.json"))
        for s in sample:
            params['sample_idx'], sample_idx = sample_idx, sample_idx+1
            file_name = str(tuple(params.values()))+'.mid'
            file_name = file_name.replace('/','_')
            file_path = os.path.join(exp_dir, file_name)
            print('saving', file_path)
            PianoRoll.from_tensor(s,normalized=True,thres = 10).to_midi(file_path)
            metadata['entries'][file_name] = params.copy()
        json_dump(metadata, os.path.join(exp_dir, "metadata.json"))

    print("sampling complete")

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


config, _ = get_config() # read config yaml at --config

def get_base_songs(path, num_songs=None, num_per_song=1, length = 32*16):
    songs = {}
    all_files = list(os.listdir(path))
    # sample num_songs songs from all_files
    if num_songs is not None:
        all_files = random.sample(all_files, num_songs)
    for f in all_files:
        f = os.path.join(path, f)
        if f.endswith('.mid') or f.endswith('.midi'):
            pr = PianoRoll.from_midi(f)
            postfix = ''
        elif f.endswith('json'):
            pr = PianoRoll.load(f)
            postfix = 'j'
        else: continue
        for i in range(num_per_song):
            clip = pr.random_slice(length)
            songs[f'{clip}{postfix} {i}'] = clip.to_tensor(0,length,padding=True,normalized=True)

    return songs

class Experiment:
    def __init__(self, exp_name, config, num_samples = 4, exp_root_dir = 'log/experiments'):
        exp_dir = os.path.join(exp_root_dir, exp_name)
        if os.path.exists(exp_dir):
            ans = input(f'Experiment {exp_name} already exists. Overwrite? (y/n)')
            if ans == 'y':
                shutil.rmtree(exp_dir)
            else:
                exit(0)
        os.makedirs(exp_dir,exist_ok=False)
        self.num_samples = num_samples
        self.exp_root_dir = exp_root_dir
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.config = config
    
    def get_param_space(self)->ParamSpace:
        pspace = ParamSpace([
            {'name': 'model', 'value_range': ['vdiff2M7'], 'relevant': True},
        ])
        return pspace

    def run_with_params(self, params, model, diffusion):
        guider = guiders.NoneGuider()
        sample_with_params(self.num_samples,self.exp_root_dir,self.exp_name,params,config,model,diffusion,guider)

    def run(self):
        print("creating model and diffusion...")
        model = create_model(config)
        model.load_state_dict(dist_util.load_state_dict(config['sampling']["model_path"], map_location="cpu"))
        diffusion = create_gaussian_diffusion(config)

        pspace = self.get_param_space()

        init_metadata(num_samples=self.num_samples, exp_root_dir=self.exp_root_dir, exp_name=self.exp_name, param_space=pspace)

        # interate through all possible combinations of parameters
        for params in pspace.all_combinations():
            print('='*20,params ,'='*20,sep='\n')
            self.run_with_params(params,model,diffusion)

class ReconstructExperiment(Experiment):
    def load_base_songs(self):
        base_songs = get_base_songs(os.path.join(self.exp_root_dir,'base_songs'), num_songs=None, num_per_song=1, length = 32*16)
        for name, bs in base_songs.items():
            file_name = name+'.mid'
            file_path = os.path.join(self.exp_dir, 'base_songs', file_name)
            print('saving', file_path)
            PianoRoll.from_tensor(bs,normalized=True,thres = 10).to_midi(file_path)
        return base_songs

    def get_param_space(self):
        self.base_songs = self.load_base_songs()
        return ParamSpace([
            {'name': 'weight', 'value_range': [1,2,4,8,16], 'relevant': True},
            {'name': 'base_song', 'value_range': list(self.base_songs.keys()), 'relevant': True},
            {'name': 'model', 'value_range': ['vdiff2M7'], 'relevant': True},
        ])

    def run_with_params(self, params, model, diffusion):
        x_a = self.base_songs[params['base_song']]
        mb = guiders.MaskBuilder(x_a)
        a_mask = mb.FirstBars(4) + mb.LastBars(4)
        #guider = guiders.ReconstructGuider(x_a,a_mask,10,diffusion.q_posterior_sample_loop)
        guider = guiders.ReconstructGuider(x_a,a_mask,params['weight'],None)
        sample_with_params(self.num_samples,self.exp_root_dir,self.exp_name,params,config,model,diffusion,guider)

class ChordExperiment(Experiment):
    def get_param_space(self) -> ParamSpace:
        return ParamSpace([
            {'name': 'weight', 'value_range': [4,6,8], 'relevant': True},
            {'name': 'cutoff_time_step', 'value_range': [0], 'relevant': True},
            {'name': 'objective_clamp', 'value_range': [0.8,0.9,1], 'relevant': True},
            {'name': 'chord progression', 'value_range':['C G Am Em F C F G', 'Am F C G'], 'relevant': True},
            {'name': 'model', 'value_range': ['vdiff2M7'], 'relevant': True},
        ])
    def run_with_params(self, params, model, diffusion):
        granularity = 32
        chord_prog = guiders.ChordGuider.generate_chroma_map(params['chord progression'], 32*16, granularity=granularity,num_repeat_interleave=32//granularity)
        guider = guiders.ChordGuider(chord_prog,mask=None,weight= params['weight'],granularity=granularity,cutoff_time_step=params['cutoff_time_step'],objective_clamp=params['objective_clamp'])
        #guider = guiders.NoneGuider()
        sample_with_params(self.num_samples,self.exp_root_dir,self.exp_name,params,config,model,diffusion,guider)

class ScratchExperiment(Experiment):
    def get_param_space(self) -> ParamSpace:
        return ParamSpace([
            {'name': 'model', 'value_range': ['vdiff2M7'], 'relevant': True},
        ])
    def run_with_params(self, params, model, diffusion):
        guider = guiders.NoneGuider()
        sample_with_params(self.num_samples,self.exp_root_dir,self.exp_name,params,config,model,diffusion,guider)

class StrokeExperiment(Experiment):
    def get_param_space(self) -> ParamSpace:
        return ParamSpace([
            {'name': 'weight', 'value_range': [4,6,8], 'relevant': True},
            #{'name': 'cutoff_time_step', 'value_range': [0], 'relevant': True},
            #{'name': 'objective_clamp', 'value_range': [0.8,0.9,1], 'relevant': True},
            {'name': 'image_file', 'value_range':['log/experiments/data/stroke3.png','log/experiments/data/stroke2.png','log/experiments/data/stroke1.png'], 'relevant': True},
            {'name': 'model', 'value_range': ['vdiff2M7'], 'relevant': True},
        ])
    def run_with_params(self, params, model, diffusion):
        guider = guiders.StrokeGuider(params['weight'],(config['decoder']['len_dec']*32,88))
        guider.load_image(params['image_file'])
        sample_with_params(self.num_samples,self.exp_root_dir,self.exp_name,params,config,model,diffusion,guider)

if __name__ == "__main__":
    dist_util.setup_dist()
    shutil.rmtree('legacy/temp/',ignore_errors=True)
    os.makedirs('legacy/temp/',exist_ok=True)
    #ReconstructExperiment('Fist4 + Last4 correct alpha',config,num_samples=4).run()
    ChordExperiment('test',config,num_samples=4).run()
    #ScratchExperiment('test',config,num_samples=4).run()
    #StrokeExperiment('Stroke',config,num_samples=4).run()

# python scripts/piano_exp.py --config config/16bar_v_scratch_lm.yaml