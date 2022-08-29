import torch
from torch.profiler import profile, record_function, ProfilerActivity

from improved_diffusion.models.transformer_unet import TransformerUnet
from utils.pianoroll import PianoRollDataset

model = TransformerUnet(256,n_blocks=2, learn_sigma= False).cuda()
d = PianoRollDataset('/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll',32)

batch_size = 128
input = [d[i]for i in range(100,100+batch_size)]
t = torch.zeros((batch_size),device='cuda')
input = torch.stack(input,0).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input,t)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))