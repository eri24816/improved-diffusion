import unittest   # The test framework
from improved_diffusion.models import transformer_unet

class Test_models(unittest.TestCase):
    def test_FFTransformer(self):
        transformer_unet.FFTransformer.test()
          

if __name__ == '__main__':
    unittest.main()