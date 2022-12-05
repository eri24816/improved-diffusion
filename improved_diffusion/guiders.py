from typing import Optional
import torch

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
    def guide(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def reset(self) -> Optional[torch.Tensor]:
        return None # return noise as z_T if needed

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

    def guide(self, z: torch.Tensor, x_pred: torch.Tensor, alpha: float) -> torch.Tensor:
        self.t-=1
        # sample z_a
        if self.q_sample_loop is not None:
            z_a = next(self.q_sample_iter)
        else:
            z_a = self.x_a

        # calculate the grad on z_b
        z.requires_grad = True
        (((x_pred - self.x_a)**2)*self.a_mask).sum().backward()
        z_b_grad = z.grad * self.b_mask

        # guide the x_b
        guided_x_b = x_pred - (self.w_r*alpha/2) * z_b_grad

        # return the guided x
        guided_x = guided_x_b * self.b_mask + z_a * self.a_mask
        if self.t==999:
            print(torch.amax(z_a,dim=(1,2)))
        if self.t == 0:
            print(torch.amax(guided_x,dim=(1,2)),torch.amax(z_a,dim=(1,2)),guided_x.shape)
        return guided_x

    def reset(self, noise) -> None:
        self.t=1000
        if self.q_sample_loop is not None:
            self.q_sample_iter = self.q_sample_loop(self.x_a,x_T=noise)

    def __repr__(self) -> str:
        return f'Exact {self.a_mask_string} {self.w_r}'

class ObjectiveGuider(Guider):
    """
    Guide the sampling process by optimizing an objective function.
    """
    def __init__(self, objective, weight = 1) -> None:
        super().__init__()
        self.objective = objective
        self.weight = weight

    def guide(self, z: torch.Tensor, x_pred: torch.Tensor, alpha: float) -> torch.Tensor:
        # calculate the grad on z_b
        z.requires_grad = True
        self.objective(x_pred).backward()
        z_b_grad = z.grad

        # guide the x_b
        guided_x_b = x_pred - (self.weight*alpha/2) * z_b_grad

        # return the guided x
        return guided_x_b


class DirectionGuider(Guider):
    """
    Guide the sampling direction with a normalized target direction.
    """
    def __init__(self, target: torch.Tensor, mask:Optional[torch.Tensor]=None, weight = 1) -> None:
        super().__init__()
        self.target = target
        self.mask = mask
        self.weight = weight

    def guide(self, z: torch.Tensor, x_pred: torch.Tensor, alpha: float) -> torch.Tensor:
        # calculate the grad on z_b
        pass



if __name__ == "__main__":
    # test the mask builder
    mb = MaskBuilder(torch.zeros(32*16,88))
    a = mb.FirstBars(2)
    b = mb.LastBars(2)
    c = mb.Upper(60)
    d = a+b*c
    # save image
    import matplotlib.pyplot as plt
    plt.imshow(d.data().T)
    plt.colorbar()
    plt.savefig('mask.png')

    print(a.data().sum(),b,c,d,sep='\n')
