import copy
import functools
import os
from turtle import xcor
from typing import Optional

import blobfile as bf
import numpy as np
import torch as th
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from improved_diffusion.gaussian_diffusion import GaussianDiffusion

from improved_diffusion.models.encoder import CyclicalKlWeight, Encoder
from improved_diffusion.resample import create_named_schedule_sampler

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import torchvision

import improved_diffusion.dist_util as dist
from torch.cuda.amp import autocast, grad_scaler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        config,model,diffusion : GaussianDiffusion,data,

        global_batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        one_bar_steps=0,
        one_bar_data=None,
    ):
        self.config = config
        self.model = model
        self.use_encoder = config['latent']['latent_size'] != 0
        if self.use_encoder:
            assert 'encoder' in model
            self.encoder : Optional[Encoder] = model['encoder']
        else:
            self.encoder = None
        self.eps_model = model['eps_model']
        self.diffusion = diffusion
        self.data = data
        if one_bar_steps > 0:
            assert one_bar_data is not None
        self.one_bar_data = one_bar_data
        self.one_bar_steps = one_bar_steps
        self.batch_size = global_batch_size//dist.get_world_size()
        self.microbatch = microbatch if microbatch > 0 else self.batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler =  create_named_schedule_sampler(schedule_sampler, diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = global_batch_size 

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self.kl_weight = CyclicalKlWeight(config['latent']['kl_weight'],int(2e4))
        self.len_dec = config['decoder']['len_dec']
        self.len_enc = config['encoder']['len_enc']
        self.latent_size = config['latent']['latent_size']

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() and dist.get_world_size() > 1:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if True:#dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if True:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.scaler = grad_scaler.GradScaler()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            if self.step + self.resume_step < self.one_bar_steps:
                assert self.one_bar_data is not None
                batch = next(self.one_bar_data)
                self.eps_model.turn_off_temporal_attn()
            else:
                batch = next(self.data)
                self.eps_model.turn_on_temporal_attn()
                
            self.model.train()
            self.run_step(batch)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % (self.save_interval//6) == 0:
                if dist.get_rank()==0:
                    print('Sampling...')
                    self.model.eval()
                    if self.len_dec !=0:
                        self.run_sample(reconstruct_target=None)
                        if self.use_encoder:
                            self.run_sample(reconstruct_target=batch.to(dist_util.dev()))
                        logger.write_img((batch[0:1].transpose(1,2).unsqueeze(1)+1)/2,'training data')
                    else:
                        self.run_sample_song()
                        logger.write_img((batch[0:1,32*32:34*32].transpose(1,2).unsqueeze(1)+1)/2,'training data')

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1


        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch,{})
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev())for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            # Run the encoder.
            if self.use_encoder:
                latent, kl_loss = self.encoder(micro,return_kl=True) # [B, n_bars, D]
                # expand the latent to fit the decoder
                expand_ratio = 32 * self.len_enc 
                latent = latent.repeat_interleave(expand_ratio, 1) # [B, L, D]
                micro_cond['condition'] = latent
            else:
                kl_loss = 0

            # Run the model. Get loss.
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.eps_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():  # type: ignore #? Why no sync?
                    losses = compute_losses()

            losses['kl_loss'] = kl_loss

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            #loss = (losses["loss"] * weights).mean()
            loss = ((losses["loss"] + losses['kl_loss'] * self.kl_weight.get(self.step)) * weights).mean()
            
            # Backprop.
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Log losses.
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

    def optimize_fp16(self):
        with autocast():
            self._log_grad_norm()
            self._anneal_lr()
            self.scaler.step(self.opt)
            self.scaler.update()
            for rate, params in zip(self.ema_rate, self.ema_params):
                update_ema(params, self.master_params, rate=rate)

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    
    def run_sample(self, reconstruct_target = None):
        # 1-2 bars model sampling
        #num_samples = max(6//self.len_dec,1)
        num_samples = 5
        model_kwargs = {}
        if self.encoder:
            if reconstruct_target is not None:
                latent = self.encoder(reconstruct_target[:num_samples],sample=False)
            else:
                latent = th.randn(num_samples,self.len_dec//self.len_enc, self.latent_size,device=dist_util.dev())
            expand_ratio = 32 * self.len_enc
            latent = latent.repeat_interleave(expand_ratio, 1)
            model_kwargs['condition'] = latent

        all_sample = self.diffusion.p_sample_loop_collect(
            self.eps_model,
            (num_samples, self.len_dec*32, 88),
            num_imgs = num_samples,
            model_kwargs = model_kwargs,
        )

        prefix = 'reconstruction/' if reconstruct_target is not None else 'generate from scrach/'
        
        def to_img(x):
            horiz_scale, vertical_scale = 4, 8
            x = x.permute(0, 1, 3, 2).contiguous()
            x = torchvision.transforms.Resize((horiz_scale*x.shape[-2],vertical_scale*x.shape[-1]),torchvision.transforms.InterpolationMode.NEAREST)(x)
            x = ((x+1)/2).clamp(0,1)
            return x

        def merge_nh(x : th.Tensor):
            n,c,h,w = x.shape
            x = x.permute(1,0,2,3).contiguous().reshape(c,n*h,w).unsqueeze(0)

            return x

        # reverse process
        sample = all_sample[:,0] 
        sample = sample.unsqueeze(1)
        
        logger.write_img(merge_nh(to_img(sample)),prefix+'reverse process')

        # sample at t = 0
        sample = all_sample[-1] 
        if reconstruct_target is not None:
            stacked = th.stack([sample,reconstruct_target[:num_samples],sample],dim=1) # stack to compare with target
            logger.write_img(merge_nh(to_img(sample)),prefix+'comparison')
        
        sample = sample.unsqueeze(1)
        logger.write_img(merge_nh(to_img(sample)),prefix+'t = 0 samples')

    def run_sample_song(self):
        # not maintained for now
        # full song model sampling
        model_kwargs = {}
        if self.encoder:
            latent = th.randn(5, self.latent_size).to(dist_util.dev())
            model_kwargs['condition'] = latent
        all_sample = self.diffusion.p_sample_loop_collect(
            self.eps_model,
            (5, 32*180, 88),
            num_imgs = 5,
            model_kwargs = model_kwargs,
        )

        sample = all_sample[:,0,32*30:32*32] # reverse process
        sample = sample.unsqueeze(1)
        sample = sample.permute(0, 1, 3, 2)
        sample = sample.contiguous()
        scale = 6
        sample = torchvision.transforms.Resize((scale*sample.shape[-2],scale*sample.shape[-1]),torchvision.transforms.InterpolationMode.NEAREST)(sample)
        logger.write_img((sample+1)/2,'reverse process')
        
        sample = all_sample[-1] # sample at t = 0
        sample = th.stack([sample[0,32*0:32*2],sample[0,32*64:32*66],sample[0,32*-2:]])
        sample = sample.unsqueeze(1)
        sample = sample.permute(0, 1, 3, 2)
        sample = sample.contiguous()
        scale = 6
        sample = torchvision.transforms.Resize((scale*sample.shape[-2],scale*sample.shape[-1]),torchvision.transforms.InterpolationMode.NEAREST)(sample)
        logger.write_img((sample+1)/2,'t = 0 samples')


    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad != None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.scaler.get_scale())

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        '''
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        '''
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
