export OPENAI_LOGDIR=/screamlab/home/eri24816/improved-diffusion/log_u2_o

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
export TRAIN_FLAGS="--lr 2e-05 --batch_size 64 --save_interval 500  --num_channels 1"