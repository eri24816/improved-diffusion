export OPENAI_LOGDIR=/screamlab/home/eri24816/improved-diffusion/log

export  MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
export DIFFUSION_FLAGS="--diffusion_steps 500 --noise_schedule cosine --use_kl True "
export TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment --save_interval 500 "

python scripts/image_train.py --data_dir datasets/mnist $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS