#export OPENAI_LOGDIR=/screamlab/home/eri24816/improved-diffusion/log_u2

#export  MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
#export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_kl True "
#export TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment --save_interval 500  --num_channels 1"

# /screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll

export OPENAI_LOGDIR=/screamlab/home/eri24816/improved-diffusion/log/2bb

export MODEL_FLAGS="--num_channels 128 --learn_sigma True --segment_length 64"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
export TRAIN_FLAGS="--lr 1e-04 --batch_size 128 --schedule_sampler loss-second-moment --save_interval 10000 --log_interval 200"