export VIDEO_DATA_PATH=/path/to/video/data
export GPUS=8  # number of GPUs
export MASTER_PORT=29704  # port for distributed training
export RUN_NAME=modelscopet2v_distillation_6_lisa1 # name of the run
export DIS_RUN_NAME=modelscopet2v_discriminator_1 # name of the run
export OUTPUT_DIR=work_dirs/$RUN_NAME  # directory to save the model checkpoints
export DIS_OUTPUT_DIR=work_dirs/$DIS_RUN_NAME

accelerate launch --num_machines 1 --num_processes $GPUS \
    --main_process_port $MASTER_PORT --mixed_precision=fp16 \
    main_m_lisa.py \
    --base_model_name=modelscope \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=256 \
    --num_frames=16 \
    --learning_rate=5e-6 \
    --loss_type="huber" \
    --adam_weight_decay=0.0 \
    --dataloader_num_workers=4 \
    --validation_steps=5000 \
    --checkpointing_steps=500 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=32 \
    --seed=453645634 \
    --enable_xformers_memory_efficient_attention \
    --report_to tensorboard wandb \
    --tracker_project_name="motion-consistency-model" \
    --tracker_run_name=$RUN_NAME \
    --dataset_path $VIDEO_DATA_PATH \
    --num_train_epochs 2 \
    --use_8bit_adam \
    --use_lora \
    --scale_lr \
    --max_grad_norm 10 \
    --lr_scheduler cosine \
    --w_min 5 \
    --w_max 15 \
    --frame_interval 8 \
    --disc_loss_type wgan \
    --disc_loss_weight 0.0 \
    --disc_learning_rate 5e-6 \
    --disc_lambda_r1 1e-5 \
    --disc_start_step 0 \
    --disc_gt_data webvid \
    --disc_tsn_num_frames 2 \
    --cd_target learn \
    --timestep_scaling_factor 4 \
    --cd_pred_x0_portion 0.5 \
    --num_ddim_timesteps 50 \
    --resume_from_checkpoint latest \
    --dis_output_dir $DIS_OUTPUT_DIR \
    --extract-code-dir /home/shaoshitong/extract_code_dir_scope_6/ \
    --lr_warmup_steps 100 --scheduler_last_step 2000 \
    --prev_train_unet /home/shaoshitong/project/mcm/work_dirs/modelscopet2v_distillation_5/checkpoint-final
