#export OPENAI_LOGDIR=/screamlab/home/eri24816/improved-diffusion/log_u2

#export  MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
#export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_kl True "
#export TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment --save_interval 500  --num_channels 1"

# /screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll

export OPENAI_LOGDIR=~/improved-diffusion/log/1b_tw

export MODEL_FLAGS="--num_channels 128 --learn_sigma True --segment_length 32"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
export TRAIN_FLAGS="--lr 2e-05 --batch_size 32 --schedule_sampler loss-second-moment --save_interval 50000 --log_interval 200 --kl_weight 0.00005"

# . env2bb.sh && python scripts/piano_train.py --data_dir /screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAG
# . env_2b_100.sh && python scripts/piano_sample.py  $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples 8 --model_path log/2bb_100/ema_0.9999_2600000.pt