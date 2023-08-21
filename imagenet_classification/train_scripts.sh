# Train a channel mode LAUDNet ResNet50 model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url /path_to_log/ \
--data_url /path_to_imagenet/ --dataset imagenet --workers 24 --config configs/finetune_100eps_1024bs_lr0x08.py \
--arch uni_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 1.0 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1 \
--mask_spatial_granularity 1-1-1-1 \
--channel_dyn_granularity 1-1-1-1 --channel_masker MLP-MLP-MLP-MLP --channel_masker_reduction 16-16-16-16 \
--channel_masker_layers 2-2-2-2 \
--dyn_mode channel-channel-channel-channel \
--dist_url tcp://127.0.0.1:20003 --print_freq 100 --round 1;


# Train a layer mode LAUDNet ResNet50 model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url /path_to_log/ \
--data_url /path_to_imagenet/ --dataset imagenet --workers 24 --config configs/finetune_100eps_1024bs_lr0x08.py \
--arch uni_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 1.0 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1 \
--mask_spatial_granularity 56-28-14-7 \
--channel_dyn_granularity 1-1-1-1 --channel_masker MLP-MLP-MLP-MLP --channel_masker_reduction 16-16-16-16 \
--channel_masker_layers 2-2-2-2 \
--dyn_mode layer-layer-layer-layer \
--dist_url tcp://127.0.0.1:20003 --print_freq 100 --round 1;


# Train a channel mode LAUDNet RegNet-Y 400M model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url /path_to_log/ \
--data_url /path_to_imagenet/ --dataset imagenet --workers 24 --config configs/finetune_100eps_1024bs_lr0x08.py \
--arch lad_regnet_y_1_6gf --finetune_from ckpts/regnet_y_1_6gf-b11a554e.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1 \
--mask_spatial_granularity 1-1-1-1 \
--channel_dyn_granularity 1-1-1-1 --channel_masker MLP-MLP-MLP-MLP --channel_masker_reduction 16-16-16-16 \
--channel_masker_layers 2-2-2-2 \
--dyn_mode channel-channel-channel-channel \
--dist_url tcp://127.0.0.1:20003 --print_freq 100 --round 1;
