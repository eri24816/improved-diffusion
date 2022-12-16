from typing import List, Optional, Union
import torch
import re
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
        pass
    def guide_x0(self, x0, *args, **kwargs) -> torch.Tensor:
        return x0
    def guide_mu(self, mu, *args, **kwargs) -> torch.Tensor:
        return mu
    def reset(self, *args, **kwargs) -> None:
        return None # return noise as z_T if needed
    def __add__(self, other):
        return CompositeGuider(self, other)

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

class ObjectiveGuider(Guider):
    """
    Guide the sampling process by optimizing an objective function.
    """
    def __init__(self, objective, weight = 0.01) -> None:
        super().__init__()
        self.objective = objective
        self.weight = weight

    def guide_x0(self, info, *args, **kwargs) -> torch.Tensor:
        x0 = info['x0']
        alpha = info['alpha']
        x0 = x0.detach()
        x0.requires_grad = True
        self.objective(x0).backward()
        guided_x = x0 - (self.weight*alpha/2) * x0.grad
        x0.requires_grad = False
        return guided_x

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

    '''
    def guide_x0(self, xt: torch.Tensor, x0: torch.Tensor, sqrt_one_minus_cum_alpha: float,t, var,*args, **kwargs ) -> torch.Tensor:
        if self.cutoff_time_step is None:
            weight_schedule = sqrt_one_minus_cum_alpha
        else:
            #weight_schedule = ((t-(1.0-self.cutoff_time_step))/self.cutoff_time_step).clamp(0,1)
            weight_schedule = sqrt_one_minus_cum_alpha*(t>self.cutoff_time_step)
        def objective(x):# cosine similarity
            # x: [batch, segment * granularity, pitch]
            x=(x+1)/2
            assert x.shape[1] == self.num_segments*self.granularity
            self.target_chroma = self.target_chroma.to(x.device)
            if self.mask is not None:
                x_chroma = to_chroma(x * self.mask)
            else:
                x_chroma = to_chroma(x)
            x_chroma = x_chroma.view(-1,self.num_segments,self.granularity,12)
            x_chroma = x_chroma.mean(2) # [batch, segment, chroma]
            # cosine similarity on chroma dimension
            cos_sim = (x_chroma*self.target_chroma).sum(2) / (1e-5+x_chroma.norm(dim=2)*self.target_chroma.norm(dim=1))
            #print(cos_sim)
            
            #print(0,cos_sim.mean().item())
            cos_sim = cos_sim.clamp(-1,self.objective_clamp)
            return torch.sum(cos_sim,dim=1) # [batch]
        
        result = guide_with_objective(x0, objective, self.mask, self.weight*weight_schedule)
        return result
    '''

    def guide_mu(self, xt: torch.Tensor, mu: torch.Tensor, sqrt_one_minus_cum_alpha: float,t, var,*args, **kwargs ) -> torch.Tensor:
        if self.cutoff_time_step is None:
            weight_schedule = sqrt_one_minus_cum_alpha
        else:
            #weight_schedule = ((t-(1.0-self.cutoff_time_step))/self.cutoff_time_step).clamp(0,1)
            weight_schedule = sqrt_one_minus_cum_alpha*(t>self.cutoff_time_step)
        def objective(x):# cosine similarity
            # x: [batch, segment * granularity, pitch]
            x=(x+1)/2
            assert x.shape[1] == self.num_segments*self.granularity
            self.target_chroma = self.target_chroma.to(x.device)
            if self.mask is not None:
                x_chroma = to_chroma(x * self.mask)
            else:
                x_chroma = to_chroma(x)
            x_chroma = x_chroma.view(-1,self.num_segments,self.granularity,12)
            x_chroma = x_chroma.mean(2) # [batch, segment, chroma]
            # cosine similarity on chroma dimension
            cos_sim = (x_chroma*self.target_chroma).sum(2) / (1e-5+x_chroma.norm(dim=2)*self.target_chroma.norm(dim=1))
            #print(cos_sim)
            
            #print(0,cos_sim.mean().item())
            cos_sim = cos_sim.clamp(-1,self.objective_clamp)
            return torch.sum(cos_sim,dim=1) # [batch]
        result = mu + get_grad(xt, objective)*var*self.weight
        return result

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



#TODO: Guiders for skyline, baseline, chord

if __name__ == "__main__":
    chord_progression = 'C F G (Am Dm)'
    chroma_map = ChordGuider.generate_chroma_map(chord_progression, 128, 16, num_repeat_interleave=2)


