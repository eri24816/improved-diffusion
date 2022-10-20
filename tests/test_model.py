import unittest   # The test framework
from improved_diffusion.models import transformer
import torch
from improved_diffusion.models.nn_utils import count_parameters

class Test_models(unittest.TestCase):
    def test_FFTransformer(self):
        #transformer.FFTransformer.test()
        instance = transformer.FFTransformer(16,frame_size=32)
        print(instance)
        count_parameters(instance)
        x = torch.randn(2,64,88)
        t = torch.randn(2)
        out = instance(x,t)
        assert out.shape == x.shape
        return out

if __name__ == '__main__':
    unittest.main()