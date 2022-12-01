import torch

class Guider:
    def __init__(self) -> None:
        pass
    def guide(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MaskedGuider(Guider):
    """
    Given a existing part of piano roll, generate the other part of the piano roll.
    Can be used for inpanting, outpainting, given prompt, or other arbitary shape of mask.
    The implementation is from the video diffusion.
    """
    def __init__(self, x_a: torch.Tensor, a_mask: torch.Tensor, w_r = 1) -> None:
        super().__init__()
        self.x_a = x_a
        self.a_mask = a_mask
        self.b_mask = 1 - a_mask
        self.w_r = w_r

    def guide(self, z: torch.Tensor, x_pred: torch.Tensor, alpha: float) -> torch.Tensor:
        # calculate the grad on z_b
        z.requires_grad = True
        (((x_pred - self.x_a)**2)*self.a_mask).sum().backward()
        z_b_grad = z.grad * self.b_mask

        # guide the x_b
        guided_x_b = x_pred - (self.w_r*alpha/2) * z_b_grad

        # return the guided x
        guided_x = guided_x_b * self.b_mask + self.x_a * self.a_mask
        return guided_x