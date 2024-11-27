#!/bin/bash
# this particular script is for evaluating the model with different entropy values for each layer
# we use the entropies that have been mentioned in the paper for SST-2..
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA="/Users/tuhinkhare/Work/GaTech-MSCS/Fall-24/CS-8803-SMR/SysMLTeam12/scripts/glue_data"

MODEL_TYPE=bert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi

# entropies = 0.01, 0.05, 0.4 -> ??

ACCURACY=75
LATENCY=100
# ENTROPIES="0,0.4,0.01,0.005,0.005,0.005,1,0,0,0,0,0"
# ENTROPIES=(
#     "1,0,0,0,0,0,0,0,0,0,0,0"
#     "0,1,0,0,0,0,0,0,0,0,0,0"
#     "0,0,1,0,0,0,0,0,0,0,0,0"
#     "0,0,0,1,0,0,0,0,0,0,0,0"
#     "0,0,0,0,1,0,0,0,0,0,0,0"
#     "0,0,0,0,0,1,0,0,0,0,0,0"
#     "0,0,0,0,0,0,1,0,0,0,0,0"
#     "0,0,0,0,0,0,0,1,0,0,0,0"
#     "0,0,0,0,0,0,0,0,1,0,0,0"
#     "0,0,0,0,0,0,0,0,0,1,0,0"
#     "0,0,0,0,0,0,0,0,0,0,1,0"
#     "0,0,0,0,0,0,0,0,0,0,0,1"
# )

# ENTROPIES=(
#     "0,0,0,0,0,0,0,0,0,0,0,0"
#     "0,0,0,0,0,0,0,0,0,0,0,1"
# )

# for ENTROPY in $ENTROPIES; do
echo $ENTROPIES
layer_idx=0
# for entropies in "${ENTROPIES[@]}"; do
python -um examples.run_highway_glue \
    --model_type $MODEL_TYPE \
    --model_name_or_path ./saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage \
    --task_name $DATASET \
    --do_eval \
    --do_lower_case \
    --data_dir $PATH_TO_DATA/$DATASET \
    --output_dir ./saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage \
    --plot_data_dir $2 \
    --max_seq_length 128 \
    --early_exit_entropy_list $1 \
    --eval_highway \
    --overwrite_cache \
    --per_gpu_eval_batch_size=1 \
    --return_per_layer_acc \
    --desired_accuracy $ACCURACY \
    --desired_latency $LATENCY \
    --eval_batchsize $3
  
  # layer_idx=$((layer_idx+1))
# done
