COUNT=0 # 1-5, 2-10, 3-15, 4-20, 5-25, 6-30
DIFFERENCE=1
END_DIFFERENCE=5

for variable  in {0..75..5}
    do
        NEW_COUNT=$[$COUNT+$DIFFERENCE]
        cp -r /home/shaoshitong/extract_code_dir_scope_${COUNT}/  /data/shaoshitong/extract_code_dir_scope_${COUNT}/
        rm -rf /home/shaoshitong/extract_code_dir_scope_${COUNT}/
        pkill python
        pkill wandb
        export VIDEO_DATA_PATH=/data/shared_data/Webvid-2M/
        export GPUS=8  # number of GPUs
        export MASTER_PORT=29501  # port for distributed training
        export RUN_NAME=modelscopet2v_extract_code_${NEW_COUNT}/  # name of the run
        export OUTPUT_DIR=work_dirs/$RUN_NAME  # directory to save the model checkpoints
        export EXPORT_DIR=/home/shaoshitong/extract_code_dir_scope_${NEW_COUNT}/
        export PREV_TRAIN_UNET=./work_dirs/modelscopet2v_distillation_${COUNT}/checkpoint-final
        export BEGIN=$variable
        export END=$[$variable+$END_DIFFERENCE]
        export DIS_OUTPUT_DIR=./work_dirs/modelscopet2v_discriminator_${COUNT}/checkpoint-discriminator-final/discriminator.pth
        
        echo $RUN_NAME
        echo $OUTPUT_DIR
        echo $EXPORT_DIR
        echo $PREV_TRAIN_UNET
        echo $BEGIN
        echo $END
        echo $DIS_OUTPUT_DIR
        
        if [ ! -d $EXPORT_DIR ];then
            bash scripts/modelscopet2v_extract_code.sh
            else
            echo "Pass modelscopet2v_extract_code"
        fi

        export MASTER_PORT=29502  # port for distributed training
        export RUN_NAME=modelscopet2v_discriminator_${NEW_COUNT}  # name of the run
        export OUTPUT_DIR=work_dirs/$RUN_NAME  # directory to save the model checkpoints
        export NEW_DIS_OUTPUT_DIR=./work_dirs/modelscopet2v_discriminator_${NEW_COUNT}/checkpoint-discriminator-final/
        if [ ! -d  $NEW_DIS_OUTPUT_DIR ];then
            bash scripts/modelscopet2v_discriminator.sh
            else
            echo "Pass modelscopet2v_discriminator"
        fi      

        export MASTER_PORT=29503  # port for distributed training
        export RUN_NAME=modelscopet2v_distillation_${NEW_COUNT}   # name of the run
        export OUTPUT_DIR=work_dirs/$RUN_NAME  # directory to save the model checkpoints
        export DIS_OUTPUT_DIR=./work_dirs/modelscopet2v_discriminator_${NEW_COUNT}/checkpoint-discriminator-final/discriminator.pth
        bash scripts/modelscopet2v_distillation_lisa.sh

        COUNT=$NEW_COUNT
    done