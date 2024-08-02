COUNT=1
DIFFERENCE=1
END_DIFFERENCE=5

for variable  in {5..15..0}
    do
        NEW_COUNT=$[$COUNT+$DIFFERENCE]
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

        export MASTER_PORT=29502  # port for distributed training
        export RUN_NAME=modelscopet2v_discriminator_${NEW_COUNT}  # name of the run
        export OUTPUT_DIR=work_dirs/$RUN_NAME  # directory to save the model checkpoints
        bash scripts/modelscopet2v_discriminator.sh        

        export MASTER_PORT=29503  # port for distributed training
        export RUN_NAME=modelscopet2v_distillation_${NEW_COUNT}   # name of the run
        export OUTPUT_DIR=work_dirs/$RUN_NAME  # directory to save the model checkpoints
        export DIS_OUTPUT_DIR=./work_dirs/modelscopet2v_discriminator_${NEW_COUNT}/checkpoint-discriminator-final/discriminator.pth
        bash scripts/modelscopet2v_distillation_lisa.sh

        COUNT=$NEW_COUNT
    done