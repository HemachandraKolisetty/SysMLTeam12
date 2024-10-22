#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=./data

MODEL_TYPE=bert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi

# ENTROPIES="0,0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7"
# # LATENCIES="2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5"
# LATENCIES="8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5 12.0 12.5 13.0 13.5 14.0"

ENTROPIES="0,0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7"
LATENCIES="4.0"

for LATENCY in $LATENCIES; do
    echo $LATENCY
    python -um examples.run_highway_glue \
        --model_type $MODEL_TYPE \
        --model_name_or_path ./saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage \
        --task_name $DATASET \
        --do_eval \
        --do_lower_case \
        --data_dir $PATH_TO_DATA/$DATASET \
        --output_dir ./saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage \
        --plot_data_dir ./plotting2_profile/ \
        --max_seq_length 128 \
        --entropies_to_evaluate $ENTROPIES \
        --eval_highway \
        --overwrite_cache \
        --per_gpu_eval_batch_size=1 \
        --return_per_layer_acc \
        --desired_latency $LATENCY \
        --path_to_lat_entropies ./plotting2_profile/saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage/lat_entropies_${LATENCY}.npy \
        --path_to_lat_n_samples ./plotting2_profile/saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage/lat_n_samples_pl.npy \
        --path_to_save_entropies ./plotting2_profile/saved_models/${MODEL_TYPE}_${MODEL_SIZE}-$DATASET-two_stage/profiled_entropies_${LATENCY}.npy
done