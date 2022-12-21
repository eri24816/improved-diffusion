from typing import List, Optional, Union
import torch
import torchvision, torch.nn.functional
import re

from torch.autograd import grad
import matplotlib.pyplot as plt

def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()

    return grads

class Plotter:
    def __init__(self, save_path='') -> None:
        self.reset()
        self.save_path = save_path
    def reset(self):
        self.lines={}
        self.fig = None
        self.ax = None

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
    def plot(self, names=None, ax=None, fig=None, **kwargs):
        if ax is None:
            if self.ax is None:
                self.fig, self.ax = plt.subplots()
            ax = self.ax
        if fig is None:
            fig = self.fig
        if names is None:
            names = self.lines.keys()
        for name in names:
            ax.plot(self.lines[name], label=name, **kwargs)
        ax.legend()
        if self.save_path:
            fig.savefig(self.save_path.replace('#',str(torch.randint(0,1000,[1]).item())))
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

    def Upper(self,pitch,piano = False):
        if piano:
            pitch -= 21
        m = torch.zeros_like(self.target_tensor)
        m[:,pitch:]=1
        return Mask(m,f'Upper{pitch}')

    def Lower(self,pitch,piano = False):
        if piano:
            pitch -= 21
        m = torch.zeros_like(self.target_tensor)
        m[:,:pitch+1]=1
        return Mask(m,f'Lower{pitch}')

    def Middle(self,pitch,piano = False):
        if piano:
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

    def set_diffusion(self, diffusion):
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

    def guide_x0(self, xt: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor,t, *args, **kwargs) -> torch.Tensor:
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

        alpha = alpha.unsqueeze(1).unsqueeze(2).expand_as(x0)

        # guide the x_b
        guided_x_b = x0 - (self.w_r*alpha/2) * z_b_grad

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
    
def to_chroma(x: torch.Tensor) -> torch.Tensor:
    chroma = []
    for i in range(12):
        chroma.append(x[...,:,i::12].sum(dim=-1))
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
    def __init__(self,objective,weight=1.) -> None:
        super().__init__()
        self.objective = objective
        self.weight = weight

        self.plotter = Plotter('legacy/temp/h_mean_std#.png')

    def guide_x0(self, xt: torch.Tensor, x0: torch.Tensor, alpha_bar: float,t, var,beta,*args, **kwargs ) -> torch.Tensor:

        # constants
        sab = alpha_bar**0.5
        s1ab = (1-alpha_bar)**0.5

        # calculate eps and mu_x0 (mu_x0 is identical to the variable x0, but I calculate it back from eps like the formula)
        eps = (xt-sab*x0.detach())/(s1ab)
        mu_x0 = (xt - s1ab*eps)/(sab)

        # calculate var_x0
        var_x0_prior = 0.1070
        var_x0 = ((1/alpha_bar-1)*var_x0_prior)/((1/alpha_bar-1)+var_x0_prior)

        # calculate g and h
        mu_x0_ = mu_x0.detach()
        mu_x0_.requires_grad = True
        g = nth_derivative(self.objective(mu_x0_), mu_x0_, n=1)
        h = nth_derivative(self.objective(mu_x0_), mu_x0_, n=2)
        h = h.clamp(max=-0.001) # must be negative to avoid nan
        h_nan = torch.isnan(h)
        h[h_nan] = -1e8

        # grad_{x_t} f(x_t) = grad_{x_t} mu_{x_0} @ grad_{mu_{x_0}} E[f(x_t)]
        #guiding_force = grad_xt_mu * (1/(1-h*var_x0)) * g
        x0.backward((1/(1-h*var_x0)) * g)
        guiding_force = xt.grad
        xt = xt.detach()

        # eps_guided = eps - w * sqrt(1-a_bar) * grad_{x_t} f(x_t)
        # from Diffusion Beats GANs
        guided_eps = eps - s1ab * guiding_force * self.weight

        guided_x0 = (xt - s1ab*guided_eps)/(sab)

        self.plotter.record('grad_mu',(1/(1-h*var_x0)) * g, 'rms',True)
        self.plotter.record('-h',-h, 'mean',True)
        self.plotter.record('g',g, 'rms',True)
        self.plotter.record('guiding_force',guiding_force, 'rms',True)
        self.plotter.record('s1ab',s1ab, 'mean',True)
        self.plotter.record('x0 std 0',x0[0], 'std',True)
        self.plotter.record('x0 std 1',x0[1], 'std',True)
        self.plotter.record('x0 std 2',x0[2], 'std',True)
        self.plotter.record('x0 std 3',x0[3], 'std',True)

        if int(t.mean().item()*1000) == 0:
            self.plotter.plot()
            #print('h_mean\th_std\tgrad_mu_rms\tguiding_force_rms\ts1ab_mean',sep='\t')

        #print(h.mean().item(),var_x0.mean().item(),grad_xt_mu.mean().item(),((guiding_force**2).sum()**0.5).item(),sep='\t')

        return guided_x0

    def reset(self, *args, **kwargs) -> None:
        self.plotter.reset()
        return super().reset(*args, **kwargs)


class ChordGuider(Guider):
    def __init__(self, target_chroma: torch.Tensor, mask:Optional[torch.Tensor]=None, weight = 0.01, granularity=16, cutoff_time_step = 0, objective_clamp = 1) -> None:
        super().__init__()
        self.target_chroma = target_chroma # [segment, chroma]
        self.mask = mask
        self.weight = weight
        self.granularity = granularity
        self.num_segments = self.target_chroma.shape[0]
        self.cutoff_time_step = cutoff_time_step
        self.objective_clamp = objective_clamp
        self.objective_guider = ObjectiveGuider(self.objective,weight=weight)
                
        if self.mask is  None:
            self.mask = 1

    def objective(self,x):# cosine similarity
        # x: [batch, segment * granularity, pitch]
        x=(x+1)/2
        assert x.shape[1] == self.num_segments*self.granularity
        self.target_chroma = self.target_chroma.to(x.device)
        
        x_chroma = to_chroma(x * self.mask)
        x_chroma = x_chroma.view(-1,self.num_segments,self.granularity,12)
        x_chroma = x_chroma.mean(2) # [batch, segment, chroma]
        # cosine similarity on chroma dimension

        cos_sim = (x_chroma*self.target_chroma).sum(2) / (1e-5+x_chroma.norm(dim=2)*self.target_chroma.norm(dim=1))
        cos_sim = cos_sim.clamp(-1,self.objective_clamp)
        return torch.sum(cos_sim) # [batch]
    
    def guide_x0(self,*args, **kwargs) -> torch.Tensor:
        guided_x0 = self.objective_guider.guide_x0(*args, **kwargs)
        return guided_x0
    
    @staticmethod
    def chord_to_chroma(chord_name:str, semi_shift=0) -> torch.Tensor:
        base_chord = chord_name[0] # C, D, E, F, G, A, B
        base_chord_ord = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}[base_chord]
        is_minor = False
        is_major7 = False
        is_minor7 = False
        is_sus2 = False
        is_sus4 = False
        is_add2 = False
        is_add4 = False
        i = 1
        while i < len(chord_name):
            if chord_name[i] == '#':
                semi_shift += 1
            elif chord_name[i] == 'b':
                semi_shift -= 1
            elif chord_name[i] == 'm':
                is_minor = True
            elif chord_name[i] == 'M':
                assert chord_name[i+1] == '7'
                is_major7 = True
                i += 1
            elif chord_name[i] == '7':
                is_minor7 = True
            elif chord_name[i:i+4] == 'sus2':
                is_sus2 = True
                i += 3
            elif chord_name[i:i+4] == 'sus4':
                is_sus4 = True
                i += 3
            elif chord_name[i:i+4] == 'add2':
                is_add2 = True
                i += 3
            elif chord_name[i:i+4] == 'add4':
                is_add4 = True
                i += 3
            else:
                raise ValueError('Unknown chord name '+chord_name)
            i += 1
        if not is_minor:
            chroma = torch.tensor([1,0,0,0,1,0,0,1,0,0,0,0])
        else:
            chroma = torch.tensor([1,0,0,1,0,0,0,1,0,0,0,0])
        if is_sus2:
            chroma[2] = 1
            chroma[3] = 0
            chroma[4] = 0
        if is_sus4:
            chroma[5] = 1
            chroma[3] = 0
            chroma[4] = 0
        if is_add2:
            chroma[2] = 1
        if is_add4:
            chroma[5] = 1
        if is_major7:
            chroma[11] = 1
        if is_minor7:
            chroma[10] = 1
        chroma = torch.roll(chroma,shifts=base_chord_ord + semi_shift,dims=-1)
        return chroma.float()

    @staticmethod
    def generate_chroma_map(chords:str, seq_length, granularity, num_repeat_interleave=1):
        '''
        example chord: 'Am F C (G E)' repeat_interleave=2
        generated chroma map: [Am, Am, F, F, C, C, G, E, Am, Am, F, F, C, C, G, E]
        '''
        chord_list = []
        for token in re.finditer(r'([A-Za-z0-9]+)|\((.*?)\)', chords):
            if token.group(1):
                chord = [token.group(1)]
                repeats = num_repeat_interleave
            else:
                print(token.group(2))
                chord = token.group(2).split(' ')
                assert num_repeat_interleave % len(chord) == 0
                repeats = num_repeat_interleave//len(chord)
            for c in chord:
                chord_list += [c]*repeats

        print(chord_list)

        assert seq_length % granularity == 0
        num_segments = seq_length // granularity
        chroma_map = []
        for i in range(num_segments):
            chroma_map.append(ChordGuider.chord_to_chroma(chord_list[i%len(chord_list)]))
        chroma_map = torch.stack(chroma_map,dim=0)
        return chroma_map

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


