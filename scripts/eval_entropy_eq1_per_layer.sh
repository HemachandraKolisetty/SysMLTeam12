#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA="./data/"

MODEL_TYPE=bert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi

# ENTROPIES="0,0.4,0.01,0.005,0.005,0.005,0.001,0.005,0.6,0.6,1,0" # 90% accuracy, 150ms latency

# ACCURACY=80
# LATENCY=120
# ENTROPIES="0,0.4,0.01,0.005,0.005,0.005,0.001,0.005,1,0,0,0"

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

ENTROPIES=(
    "0,0,0,0,0,0,0,0,0,0,0,0"
    "0,0,0,0,0,0,0,0,0,0,0,1"
)

# for ENTROPY in $ENTROPIES; do
echo $ENTROPIES
layer_idx=0
for entropies in "${ENTROPIES[@]}"; do
  python -um examples.run_highway_glue \
      --model_type $MODEL_TYPE \
      --model_name_or_path ./saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage \
      --task_name $DATASET \
      --do_eval \
      --do_lower_case \
      --data_dir $PATH_TO_DATA/$DATASET \
      --output_dir ./saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage \
      --plot_data_dir ./plotting-per-layer-3-tk/${layer_idx} \
      --max_seq_length 128 \
      --early_exit_entropy_list $entropies \
      --eval_highway \
      --overwrite_cache \
      --per_gpu_eval_batch_size=1 \
      --return_per_layer_acc \
      --desired_accuracy $ACCURACY \
      --desired_latency $LATENCY 
  
  layer_idx=$((layer_idx+1))
done
