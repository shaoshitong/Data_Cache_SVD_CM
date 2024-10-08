accelerate launch --num_machines 1 --num_processes $GPUS \
    --main_process_port $MASTER_PORT --mixed_precision=fp16 \
    extract_dataset_m.py \
    --base_model_name=modelscope \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=256 \
    --num_frames=16 \
    --learning_rate=5e-6 \
    --loss_type="huber" \
    --adam_weight_decay=0.0 \
    --dataloader_num_workers=2 \
    --validation_steps=5000 \
    --checkpointing_steps=500 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --seed=$SEED \
    --enable_xformers_memory_efficient_attention \
    --report_to tensorboard wandb \
    --tracker_project_name="motion-consistency-model" \
    --tracker_run_name=$RUN_NAME \
    --dataset_path $VIDEO_DATA_PATH \
    --num_train_epochs 1 \
    --use_8bit_adam \
    --use_lora \
    --scale_lr \
    --max_grad_norm 10 \
    --lr_scheduler cosine \
    --w_min 5 \
    --w_max 10 \
    --frame_interval 8 \
    --disc_loss_type wgan \
    --disc_loss_weight 1 \
    --disc_learning_rate 5e-5 \
    --disc_lambda_r1 1e-5 \
    --disc_start_step 0 \
    --disc_gt_data webvid \
    --disc_tsn_num_frames 16 \
    --cd_target learn \
    --timestep_scaling_factor 4 \
    --cd_pred_x0_portion 0.5 \
    --num_ddim_timesteps 50 \
    --resume_from_checkpoint latest \
    --extract-code-dir $EXPORT_DIR \
    --web-dataset-end $END \
    --web-dataset-begin $BEGIN \
    --prev_train_unet $PREV_TRAIN_UNET \
    --prev_teacher_unet $PREV_TEACHER_UNET
