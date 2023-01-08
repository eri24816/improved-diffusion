from typing import List, Optional, Union
import torch
import torchvision, torch.nn.functional
import re

from torch.autograd import grad
import matplotlib.pyplot as plt
import einops
from improved_diffusion.gaussian_diffusion import GaussianDiffusion

from utils import music

def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()

    return grads

def pr2img(pr,path):
    pr = pr.detach().cpu()
    pr = einops.rearrange(pr,'t p -> p t')
    pr = (pr+1)/2
    pr = pr.flip(0)
    # use plt to save
    plt.figure(figsize=(10,10))
    plt.imshow(pr,vmin=0,vmax=1)
    #plt.colorbar()
    plt.savefig(path)
    plt.close()

class Plotter:
    def __init__(self, save_path='') -> None:
        self.reset()
        self.save_path = save_path
        self.title = ''
        self.fig = None
        self.ax = None
    def reset(self):
        self.lines={}
    def set_title(self, title):
        self.title = title

    def record(self, name, value, reduce='mean', log_scale=False):
        if name not in self.lines:
            self.lines[name] = []
        if isinstance(value, torch.Tensor):
            if reduce == 'mean':
                value = value.mean()
            elif reduce == 'sum':
                value = value.sum()
            elif reduce == 'max':
                value = value.max()
            elif reduce == 'min':
                value = value.min()
            elif reduce == 'std':
                value = value.std()
            elif reduce == 'var':
                value = value.var()
            elif reduce == 'rms':
                value = value.pow(2).mean().sqrt()
            elif reduce == 'norm':
                value = value.norm()
            if log_scale:
                value = torch.log10(value)
        self.lines[name].append(value.item())
    def plot(self, names=None, **kwargs):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            ax = self.ax
            fig = self.fig
        if names is None:
            names = self.lines.keys()
        for name in names:
            ax.plot(self.lines[name], label=name, **kwargs)
        ax.legend()
        ax.set_title(self.title)
        if self.save_path:
            fig.savefig(self.save_path.replace('#',self.title))
        return fig, ax
        

class Mask:
    def __init__(self, data:torch.Tensor, name:str):
        self._data = data.bool()
        self.name = name
    
    def data(self):
        return self._data

    def __repr__(self):
        return f"Mask({self.name})"

    '''
    operations
    '''

    def __add__(self, other):
        return Mask(self.data() + other.data(), f"{self.name}+{other.name}")

    def __sub__(self, other):
        return Mask(self.data() - other.data(), f"({self.name})-({other.name})")

    def __mul__(self, other):
        return Mask(self.data() * other.data(), f"({self.name})*({other.name})")

    def __truediv__(self, other):
        return Mask(self.data() / other.data(), f"({self.name})/({other.name})")

    def __and__(self, other):
        return Mask(self.data() & other.data(), f"{self.name}&{other.name}")

    def __or__(self, other):
        return Mask(self.data() | other.data(), f"{self.name}|{other.name}")

    def __xor__(self, other):
        return Mask(self.data() ^ other.data(), f"{self.name}^{other.name}")

    def __invert__(self):
        return Mask(~self.data(), f"~{self.name}")

    def __eq__(self, other):
        return Mask(self.data() == other.data(), f"{self.name}=={other.name}")

class MaskBuilder:
    """
    A helper for building masks
    """
    def __init__(self,target_tensor) -> None:
        self.target_tensor = target_tensor

    def First(self,n):
        m = torch.zeros_like(self.target_tensor)
        m[:n]=1
        # to string 3 digits after the decimal point
        return Mask(m,f'First{n/32}')

    def FirstBars(self,n):
        return self.First(n*32)

    def Last(self,n):
        m = torch.zeros_like(self.target_tensor)
        m[-n:]=1
        return Mask(m,f'Last{n/32}')

    def LastBars(self,n):
        return self.Last(n*32)

    def Bar(self,n,end_bar=-1):
        m = torch.zeros_like(self.target_tensor)
        if end_bar == -1:
            end_bar = n+1
        m[int(n*32):int(end_bar*32)]=1
        return Mask(m,f'Bar{n}-{end_bar}')

    def Upper(self,pitch,midi = False):
        if midi:
            pitch -= 21
        m = torch.zeros_like(self.target_tensor)
        m[:,pitch:]=1
        return Mask(m,f'Upper{pitch}')

    def Lower(self,pitch,midi = False):
        if midi:
            pitch -= 21
        m = torch.zeros_like(self.target_tensor)
        m[:,:pitch+1]=1
        return Mask(m,f'Lower{pitch}')

    def Middle(self,pitch,midi = False):
        if midi:
            pitch -= 21
        m = torch.zeros_like(self.target_tensor)
        m[:,pitch:]=1
        m[:,:pitch+1]=1
        return Mask(m,f'Middle{pitch}')


class Guider:
    def __init__(self) -> None:
        self.diffusion = None
    def guide_x0(self, x0, *args, **kwargs) -> torch.Tensor:
        return x0
    def guide_mu(self, mu, *args, **kwargs) -> torch.Tensor:
        return mu
    def reset(self, *args, **kwargs) -> None:
        return None # return noise as z_T if needed
    def get_timestep_range_and_noise(self):
        '''
        Returns (min_t, max_t, noise)
        '''
        return 0, None, None
    def __add__(self, other):
        return CompositeGuider(self, other)

    def set_diffusion(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

class CompositeGuider(Guider):
    def __init__(self, *guiders) -> None:
        super().__init__()
        self.guiders = guiders
    def guide_x0(self, x0, *args, **kwargs) -> torch.Tensor:
        for guider in self.guiders:
            x0 = guider.guide_x0(x0, *args, **kwargs)
        return x0
    def guide_mu(self, mu, *args, **kwargs) -> torch.Tensor:
        for guider in self.guiders:
            mu = guider.guide_mu(mu, *args, **kwargs)
        return mu
    def reset(self, *args, **kwargs) -> None:
        for guider in self.guiders:
            guider.reset(*args, **kwargs)
    def __repr__(self) -> str:
        return f"CompositeGuider({self.guiders})"

class NoneGuider(Guider):
    def __repr__(self) -> str:
        return ''

class ReconstructGuider(Guider):
    """
    Given a existing part of piano roll, generate the other part of the piano roll.
    Can be used for inpanting, outpainting, given prompt, or other arbitary shape of mask.
    The implementation is from the video diffusion.
    """
    def __init__(self, x_a: torch.Tensor, a_mask: Mask, w_r = 1, q_sample_loop = None) -> None:
        super().__init__()
        self.x_a = x_a
        self.a_mask = a_mask.data()
        self.b_mask = ~ a_mask.data()
        self.a_mask_string = str(a_mask)
        self.w_r = w_r
        self.q_sample_loop = q_sample_loop
        self.q_sample_iter = None

    def guide_x0(self, xt: torch.Tensor, x0: torch.Tensor, alpha_bar: torch.Tensor,t, *args, **kwargs) -> torch.Tensor:
        self.t-=1
        self.x_a = self.x_a.to(xt.device)
        self.a_mask = self.a_mask.to(xt.device)
        self.b_mask = self.b_mask.to(xt.device)
        # sample z_a
        if self.q_sample_loop is not None:
            z_a = next(self.q_sample_iter)  
        else:
            z_a = self.x_a

        # calculate the grad on z_b
        xt.requires_grad = True
        (((x0 - self.x_a)**2)*self.a_mask).sum().backward()
        z_b_grad = xt.grad * self.b_mask

        # guide the x_b
        guided_x_b = x0 - (self.w_r*alpha_bar**0.5/2) * z_b_grad

        # return the guided x
        guided_x = guided_x_b * self.b_mask + z_a * self.a_mask
        return guided_x

    def reset(self, noise) -> None:
        self.t=1000
        if self.q_sample_loop is not None:
            self.q_sample_iter = self.q_sample_loop(self.x_a,x_T=noise)

    def __repr__(self) -> str:
        return f'Recons {self.a_mask_string} {self.w_r}'

def guide_with_objective(x, objective, mask,weight = 0.01):
    x = x.detach()
    x.requires_grad = True
    if mask is None:
        mask = torch.ones_like(x)
    o = (objective(x) * weight).sum()
    o.backward()
    guided_x = x + mask * x.grad
    #print((mask * weight * x.grad).norm())
    return guided_x.detach()

def get_grad(x, objective):
    x = x.detach()
    x.requires_grad = True
    o = objective(x).sum()
    o.backward()
    return x.grad

def cosine_similarity(x:torch.Tensor, target_direction, mask, weight = 0.01):
    return torch.sum((x*target_direction).sum(1,2) / (x.norm()*target_direction.norm()))   
    
def to_chroma(x: torch.Tensor, L=1) -> torch.Tensor:
    chroma = []
    if torch.isnan(x).any():
            print('nan x')
            print(x)
    for i in range(12):
        chroma.append((x[...,:,i::12]**L).sum(dim=-1)**(1/L))
        if torch.isnan(chroma[-1]).any():
            print('nan')
            print(x[...,:,i::12])
            print((x[...,:,i::12]**L).mean(dim=-1))
            exit()
    chroma = torch.stack(chroma,dim=-1)
    chroma = torch.roll(chroma,shifts=-3,dims=-1)
    return chroma

class DirectionGuider(Guider):
    """
    Guide the sampling direction with a normalized target direction.
    """
    def __init__(self, target: torch.Tensor, mask:Optional[torch.Tensor]=None, weight = 0.01) -> None:
        super().__init__()
        self.target = target
        self.mask = mask
        self.weight = weight
        def objective(x):# cosine similarity
            return (x*self.target).sum() / (x.norm()*self.target.norm()) * self.mask            

    def guide_x0(self, xt: torch.Tensor, x0: torch.Tensor, weight: float, *args, **kwargs) -> torch.Tensor:
        # calculate the grad on z_b
        pass

class ObjectiveGuider(Guider):
    '''
    Guide the sampling direction with a objective function.
    '''
    def __init__(self,objective,weight=1.,use_ddim=False) -> None:
        super().__init__()
        self.objective = objective
        self.weight = weight

        self.plotter = Plotter('legacy/temp/h_mean_std#.png')
        self.use_ddim = use_ddim # x0 or mu

    def guide_x0(self, xt: torch.Tensor, x0: torch.Tensor, alpha_bar: float,t, var,beta,*args, **kwargs ) -> torch.Tensor:
        if not self.use_ddim:
            return x0
        # constants
        sab = alpha_bar**0.5
        s1ab = (1-alpha_bar)**0.5

        # calculate eps and mu_x0 (mu_x0 is identical to the variable x0, but I calculate it back from eps like the formula)
        eps = (xt-sab*x0.detach().clamp(-1,1))/(s1ab)
        

        # calculate var_x0
        var_x0_prior = 0.1070
        var_x0 = ((1/alpha_bar-1)*var_x0_prior)/((1/alpha_bar-1)+var_x0_prior)

        # calculate g and h
        mu_x0_ = x0.detach().clamp(-1,1)
        mu_x0_.requires_grad = True
        g = nth_derivative(self.objective(mu_x0_), mu_x0_, n=1)
        h = nth_derivative(self.objective(mu_x0_), mu_x0_, n=2)
        h = h.clamp(max=-0.001) # must be negative to avoid nan
        h_nan = torch.isnan(h)
        h[h_nan] = -1e8

        # grad_{x_t} f(x_t) = grad_{x_t} mu_{x_0} @ grad_{mu_{x_0}} E[f(x_t)]
        #grad_xt_mu = nth_derivative(x0.sum(),xt,1)
        #guiding_force = grad_xt_mu * (1/(1-h*var_x0)) * g
        #x0.backward((1/(1-h*var_x0)) * g)
        #guiding_force = xt.grad
        guiding_force = 0.4*(1/(1-h*var_x0)) * g
        xt = xt.detach()

        # eps_guided = eps - w * sqrt(1-a_bar) * grad_{x_t} f(x_t)
        # from Diffusion Beats GANs
        guided_eps = eps - s1ab * guiding_force * self.weight

        guided_x0 = (xt - s1ab*guided_eps)/(sab).clamp(-1,1)

        self.plotter.record('grad_mu',(1/(1-h*var_x0)) * g, 'rms',True)
        self.plotter.record('-h',-h, 'mean',True)
        self.plotter.record('g',g, 'rms',True)
        self.plotter.record('guiding_force',guiding_force, 'rms',True)
        self.plotter.record('s1ab',s1ab, 'mean',True)
        self.plotter.record('x0 std 0',x0[0], 'std',True)
        self.plotter.record('x0 std 1',x0[1], 'std',True)
        self.plotter.record('x0 std 2',x0[2], 'std',True)
        self.plotter.record('x0 std 3',x0[3], 'std',True)
        self.plotter.record('objective',self.objective(mu_x0_)/4/16, 'mean')
        self.plotter.record('objective guided',self.objective(guided_x0)/4/16, 'mean')

        if int(t.mean().item()*1000)%100 == 0:
            self.plotter.plot()
            #print('h_mean\th_std\tgrad_mu_rms\tguiding_force_rms\ts1ab_mean',sep='\t')
            pr2img(mu_x0_[0,:32*4],f'log/experiments/temp/x0_{int(t.mean().item()*1000)}.png')
            pr2img(guided_x0[0,:32*4],f'log/experiments/temp/guided_x0_{int(t.mean().item()*1000)}.png')
            pr2img(xt[0,:32*4],f'log/experiments/temp/guided_xt_{int(t.mean().item()*1000)}.png')

        if int(t.mean().item()*1000) == 0:
            self.plotter.reset()

        #print(h.mean().item(),var_x0.mean().item(),grad_xt_mu.mean().item(),((guiding_force**2).sum()**0.5).item(),sep='\t')

        return guided_x0
    
    def guide_mu(self, xt: torch.Tensor, x0: torch.Tensor ,mu, alpha_bar: float,t, var,beta,*args, **kwargs ) -> torch.Tensor:
        if self.use_ddim:
            return mu.detach()
        
        if t.mean().item() < 0.05:
            return mu.detach()
        # constants
        sab = alpha_bar**0.5
        s1ab = (1-alpha_bar)**0.5

        # calculate eps and mu_x0 (mu_x0 is identical to the variable x0, but I calculate it back from eps like the formula)
        eps = (xt-sab*x0.detach().clamp(-1,1))/(s1ab)
        

        # calculate var_x0
        var_x0_prior = 0.1070
        var_x0 = ((1/alpha_bar-1)*var_x0_prior)/((1/alpha_bar-1)+var_x0_prior)

        # calculate g and h
        #mu_x0_ = x0.detach().clamp(-1,1)
        mu_x0_ = x0.detach()

        mu_x0_.requires_grad = True
        g = nth_derivative(self.objective(mu_x0_), mu_x0_, n=1)
        h = nth_derivative(self.objective(mu_x0_), mu_x0_, n=2)
        h = h.clamp(max=-0.001) # must be negative to avoid nan
        h_nan = torch.isnan(h)

        # grad_{x_t} f(x_t) = grad_{x_t} mu_{x_0} @ grad_{mu_{x_0}} E[f(x_t)]
        #grad_xt_mu = nth_derivative(x0.sum(),xt,1)
        #guiding_force = grad_xt_mu * (1/(1-h*var_x0)) * g
        #x0 = x0.clamp(-1,1)

        grad_x0_sim = (1/(1-h*var_x0)) * g
        #grad_x0_sim[(grad_x0_sim<0) & (x0<=-1)] = 0 # simulate clamp
        #grad_x0_sim[(grad_x0_sim>0) & (x0>=1)] = 0
        grad_x0_sim[h_nan] = 0
        x0.backward(grad_x0_sim)
        guiding_force = xt.grad

        #guiding_force = 0.4*(1/(1-h*var_x0)) * g

        # mu_guided = mu + var* grad_{x_t} f(x_t)
        # from Diffusion Beats GANs
        guided_mu = mu + var * guiding_force * self.weight

        self.plotter.record('guiding_force',guiding_force, 'rms',True)
        self.plotter.record('s1ab',s1ab, 'mean',True)
        self.plotter.record('mean_var',var, 'mean',True)
        if int(t.mean().item()*1000)%100 == 0:
            self.plotter.plot()
            #print('h_mean\th_std\tgrad_mu_rms\tguiding_force_rms\ts1ab_mean',sep='\t')
            pr2img(mu_x0_[0,:32*4],f'log/experiments/temp/x0_{int(t.mean().item()*1000)}.png')
            pr2img(guided_mu[0,:32*4],f'log/experiments/temp/guided_x0_{int(t.mean().item()*1000)}.png')
            pr2img(xt[0,:32*4],f'log/experiments/temp/guided_xt_{int(t.mean().item()*1000)}.png')

        if int(t.mean().item()*1000) == 0:
            self.plotter.reset()

        return guided_mu

    def reset(self, *args, **kwargs) -> None:
        self.plotter.reset()
        return super().reset(*args, **kwargs)


class ChordGuider(Guider):
    def __init__(self, chord_sequence: str, mask:Optional[torch.Tensor]=None, weight = 0.01,num_repeat_interleave=1, granularity=16, num_segments=32, cutoff_time_step = 0, objective_clamp = 1,use_ddim = False,gamma=1) -> None:
        super().__init__()
        self.target_chord = music.chord_sequence_to_chords(chord_sequence,num_repeat_interleave) # [segment]
        self.target_chroma = music.generate_chroma_map(self.target_chord,num_segments) # [segment, chroma]
        self.mask = mask
        self.weight = weight
        self.granularity = granularity
        self.num_segments = self.target_chroma.shape[0]
        self.cutoff_time_step = cutoff_time_step
        self.objective_clamp = objective_clamp
        #self.objective_guider = ObjectiveGuider(self.objective,weight=weight)
        self.objective_guider = ObjectiveGuider(self.objective_softmax,weight=weight,use_ddim=use_ddim)
                
        if self.mask is  None:
            self.mask = 1

        self.classes_chord = map(music.Chord,['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm'])
        self.classes_chroma = torch.stack([c.to_chroma() for c in self.classes_chord],0) # [class, chroma]
        self.target_idx = [c.base_chord_ord+c.is_minor*12 for c in self.target_chord] # [segment]
        self.target_idx*= (num_segments//len(self.target_idx))
        print([['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm'][i] for i in self.target_idx])

        target_chroma = self.target_chroma.repeat_interleave(self.granularity,dim=0) #[tick, chroma]
        target_pr = target_chroma.repeat(1,10) # [tick, C-1 - B9]
        self.target_pr = target_pr[:,9:9+88]
        self.gamma = gamma

    def objective(self,x):# cosine similarity
        # x: [batch, segment * granularity, pitch]
        x=(x+1)/2
        assert x.shape[1] == self.num_segments*self.granularity
        self.target_chroma = self.target_chroma.to(x.device)
        
        L = 1
        x_chroma = to_chroma(x * self.mask,L)
        x_chroma = x_chroma.view(-1,self.num_segments,self.granularity,12)

        x_chroma = (x_chroma**L).mean(2)**(1/L) # [batch, segment, chroma]
        y_chroma = self.target_chroma
        # Transform before cosine similarity
        
        x_chroma = torch.log(x_chroma/(x_chroma.max(dim=2,keepdim=True).values.detach()+1e-8)+1)
        #y_chroma = torch.log(y_chroma+1)
        
        #x_chroma = torch.sigmoid(x_chroma)
        #y_chroma = torch.sigmoid(y_chroma)

        smooth = 1e-5
        cos_sim = ((x_chroma*self.target_chroma).sum(2)) / (smooth+x_chroma.norm(dim=2)*self.target_chroma.norm(dim=1))
        cos_sim = cos_sim.clamp(-1,self.objective_clamp)
        return torch.sum(cos_sim) # [batch]

    def objective_log(self,x):
        # x: [batch, segment * granularity, pitch]
        x=(x+1)/2
        assert x.shape[1] == self.num_segments*self.granularity
        self.target_chroma = self.target_chroma.to(x.device)
        
        L = 1
        x_chroma = to_chroma(x * self.mask,L)
        x_chroma = x_chroma.view(-1,self.num_segments,self.granularity,12)

        x_chroma = (x_chroma**L).mean(2)**(1/L) # [batch, segment, chroma]
        x_chroma /= x_chroma.sum(dim=2,keepdim=True)
        target_chroma = self.target_chroma  # [segment, chroma]
        #target_chroma /= target_chroma.sum(dim=1,keepdim=True)
        
        objective = target_chroma*torch.log(x_chroma+1e-5)# + (1-target_chroma)*torch.log(1-x_chroma+1e-5)
        
        return torch.sum(objective) # [batch]

    def objective_softmax(self,x): # one layer classifier
        # x: [batch, segment * granularity, pitch]
        x=(x+1)/2
        assert x.shape[1] == self.num_segments*self.granularity
        self.target_chroma = self.target_chroma.to(x.device)
        self.target_chroma = self.target_chroma / self.target_chroma.norm(dim=1,keepdim=True)
        
        L = 1
        x_chroma = to_chroma(x * self.mask,L)
        x_chroma = x_chroma.view(-1,self.num_segments,self.granularity,12)
        x_chroma = (x_chroma**L).mean(2)**(1/L) # [batch, segment, chroma]
        x_chroma = x_chroma / x_chroma.norm(dim=2,keepdim=True)
        #x_chroma = torch.log(x_chroma+1).clamp(0)

        # dot product similarity on chroma dimension

        classes_chroma = self.classes_chroma.to(x.device) # [class=24, chroma=12]
        classes_chroma = classes_chroma / classes_chroma.norm(dim=1,keepdim=True)

        classes_sim = einops.einsum(x_chroma,classes_chroma,'batch segment chroma, class chroma -> batch segment class') # [batch, segment, class=24]
        for i, target_idx in enumerate(self.target_idx):
            classes_sim[:,i,target_idx] = -1e8
            #classes_sim[:] = -1

        target_sim = einops.einsum(x_chroma,self.target_chroma,'batch segment chroma, segment chroma -> batch segment').unsqueeze(-1) # [batch, segment, class=1]

        sim = torch.cat([classes_sim,target_sim],dim=2) # [batch, segment, class=25]

        # softmax and return log prob of target class

        distribution = torch.nn.functional.softmax(sim*self.gamma,dim=2) # [batch, segment, class_prob=25]

        target_prob = distribution[:,:,-1] # [batch, segment]

        log_prob = torch.log(target_prob+1e-8)
        
        return torch.sum(log_prob)
        
    
    def guide_x0(self,*args, **kwargs) -> torch.Tensor:
        guided_x0 = self.objective_guider.guide_x0(*args,**kwargs)
        return guided_x0

    def guide_mu(self,*args, **kwargs) -> torch.Tensor:
        guided_mu = self.objective_guider.guide_mu(*args,**kwargs)
        return guided_mu

    # def get_timestep_range_and_noise(self):
    #     w=0.7
    #     return 0,900,self.diffusion.q_sample((self.target_pr*2-1)*w+torch.randn_like(self.target_pr)*(1-w),torch.tensor(900))

class StrokeGuider(Guider):
    def __init__(self, weight, shape, blur=1, start_timestep = None, density_tolerance=0) -> None:
        super().__init__()
        self.objective_guider = ObjectiveGuider(self.objective, weight=weight)
        self.img = None
        self.shape = shape
        self.blur = blur

        # 15*15 gaussian kernel
        x,y = torch.meshgrid(torch.linspace(-11,11,23),torch.linspace(-4,4,9))
        self.gaussian_kernel = torch.exp(-(x**2/10+y**2)/2/15).float()
        self.gaussian_kernel = self.gaussian_kernel / torch.sum(self.gaussian_kernel)

        self.threshold = 0.3
        self.save_img = False
        self.start_timestep = start_timestep

    def load_image(self,path):
        raw_img = torchvision.io.read_image(path).float()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.permute(0,2,1)),
            torchvision.transforms.Resize(self.shape),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Lambda(lambda x: x.flip(2)),
            torchvision.transforms.GaussianBlur(int((self.blur*5)/2)*2+1, sigma=(self.blur, self.blur)),
            torchvision.transforms.Lambda(lambda x: x[0]/256),
        ])
        self.img = transform(raw_img)*0.05
    
    def objective(self,x):
        self.img = self.img.to(x.device)
        self.gaussian_kernel = self.gaussian_kernel.to(x.device)
        x = ((x+1)/2).clamp(0,self.threshold)/self.threshold
        x = torch.nn.functional.conv2d(x.unsqueeze(1),self.gaussian_kernel.unsqueeze(0).unsqueeze(0),padding=(self.gaussian_kernel.shape[0]//2,self.gaussian_kernel.shape[1]//2)).squeeze(1)

        if self.save_img is not False:
            import matplotlib.pyplot as plt
            fname = f'legacy/temp/{self.save_img}.png'
            plt.imsave(fname, torch.cat([x[0].detach(),self.img.detach()],dim=1).cpu().numpy())
            self.save_img = False
        
        return -torch.sum((x-self.img)**2)

    def guide_x0(self,t,*args, **kwargs) -> torch.Tensor:
        if int(t.mean()*1000)%100==0:
            self.save_img = int(t.mean()*1000)
        else:
            self.save_img = False
        guided_x0 = self.objective_guider.guide_x0(*args,t=t, **kwargs)
        return guided_x0

    def get_timestep_range_and_noise(self):
        if self.start_timestep is None:
            return super().get_timestep_range_and_noise()
        min_timestep = 0
        max_timestep = self.start_timestep
        noise = self.diffusion.q_sample((self.img*2-1),torch.ones([self.img.shape[0],self.img.shape[1]],dtype=torch.long)* max_timestep)
        fname = f'legacy/temp/start.png'
        import matplotlib.pyplot as plt
        plt.imsave(fname, noise.cpu().numpy())

        return min_timestep, max_timestep, noise

#TODO: Guiders for skyline, baseline, chord

if __name__ == "__main__":
    chord_progression = 'C F G (Am Dm)'
    chroma_map = ChordGuider.generate_chroma_map(chord_progression, 128, 16, num_repeat_interleave=2)


