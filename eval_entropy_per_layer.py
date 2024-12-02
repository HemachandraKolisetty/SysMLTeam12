'''
Script for running per layer experiments with entropy thresholds given in the paper
Entropy threshold selected - 0, 0.01, 0.05, 0.4
'''
import sys
import os
import subprocess
import time

# PATH_TO_DATA = "/Users/tuhinkhare/Work/GaTech-MSCS/Fall-24/CS-8803-SMR/SysMLTeam12/scripts/glue_data"
# MODEL_TYPE="bert"  # bert or roberta
# MODEL_SIZE="base"  # base or large
# DATASET="SST-2"  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
# MODEL_NAME = f"{MODEL_TYPE}-{MODEL_SIZE}"

# if MODEL_TYPE == "bert":
#     MODEL_NAME = f"{MODEL_TYPE}-{MODEL_SIZE}-uncased"

BASE_PATH = "/home/hice1/epiper7/Documents/SysML/SysMLTeam12"

def run_experiment(entropy_config, out_dir, batch_size = 1):
    # env = os.environ.copy()
    # env['PYTHONPATH'] = f"{BASE_PATH}:{env.get('PYTHONPATH', '')}"
    
    command = f"./scripts/eval_entropy_exp2_per_layer_parameterised.sh {entropy_config} {out_dir} {batch_size}"
    print(f"Command: {command}")
    subprocess.run(command, shell=True, check=True)


def main():
    num_layers = 12
    base_entropy_thresholds = [0.01]
    base_entropy_thresholds.reverse()
    for e_thresh in base_entropy_thresholds:
        for layer in range(num_layers):
            out_dir = f"{BASE_PATH}/per-layer-experiment-3/ethresh_{e_thresh}/layer_{layer+1}"
            os.makedirs(out_dir, exist_ok=True)
            force_exit_value = 1
            if layer == num_layers - 1:
                force_exit_value = 0
            entropy_config = ([e_thresh] * layer + [force_exit_value] + [0] * (num_layers - layer - 1))
            print(f"LOG::Running experiment with entropy config: {entropy_config} - output dir: {out_dir}")
            run_experiment(",".join([str(val) for val in entropy_config]), out_dir)
            print(f"== END ==")
        time.sleep(10)


def get_batch_latencies():
    num_layers = 12
    base_entropy_thresholds = [0]
    e_thresh = 0
    batch_sizes = [1, 2, 4, 8, 16]
    base_entropy_thresholds.reverse()
    for batch in batch_sizes:
        for layer in range(num_layers):
            out_dir = f"{BASE_PATH}/per-layer-experiment-4/batch_{batch}/layer_{layer+1}"
            os.makedirs(out_dir, exist_ok=True)
            force_exit_value = 1
            if layer == num_layers - 1:
                force_exit_value = 0
            entropy_config = ([e_thresh] * layer + [force_exit_value] + [0] * (num_layers - layer - 1))
            print(f"LOG::Running experiment with entropy config: {entropy_config} - output dir: {out_dir}")
            run_experiment(",".join([str(val) for val in entropy_config]), out_dir, batch_size=batch)
            print(f"== END ==")
        time.sleep(10)

get_batch_latencies()
# main()