#!/bin/bash

# GPT-OSS 120B Training Script
# OpenAI's GPT-OSS model with 36 layers, 128 experts, YARN RoPE, and sliding window attention
#
ulimit -n 1048576

# For rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# Will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Load model configuration
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/gptoss-20B.sh"

# Checkpoint configuration
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/gpt-oss-20b
   --ref-load $BASE_DIR/gpt-oss-20b_torch_dist
   --save /root/saved_model/
   --save-interval 1
   --no-save-optim
   --no-save-rng
)

# Rollout configuration for RL data generation
ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 2
   --rollout-batch-size 8
   --n-samples-per-prompt 1
   --rollout-max-response-len 6000
   --rollout-temperature 0.8

   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 256
   --balance-data
   --rollout-stop-token-ids 151329 151336 151338  # Adjust for GPT-OSS tokenizer
)

# Evaluation configuration
EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data aime $BASE_DIR/rl_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 6000
   --eval-top-p 0.7
)

# Performance and parallelism configuration
# Adjust these based on your hardware setup
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 10000
)

# GRPO/GSPO configuration
GRPO_ARGS=(
   --advantage-estimator gspo
   #--use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 1e-4
   --eps-clip-high 2e-4

   --use-tis
)

# Optimizer configuration
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# Wandb logging
WANDB_ARGS=(
   # --use-wandb
   # --wandb-project gpt-oss-120B
   # --wandb-group gpt-oss-test-run
   # --wandb-key ${WANDB_KEY}
)

# SGLang configuration for inference
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-data-parallel-size 1
   --sglang-disable-cuda-graph
   --sglang-disable-radix-cache
)

# Miscellaneous configuration
MISC_ARGS=(
   # Performance optimizations
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32

   # Attention backend
   --attention-backend flash
)

# Launch the master node of ray in container
export MASTER_ADDR=${MLP_WORKER_0_HOST}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Start Ray workers on other nodes
# for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
#   if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
#     continue
#   fi
#   echo "Starting Ray worker on ${WORKER_IP}"
#   ssh root@"${WORKER_IP}" \
#     "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
# done
# wait

# Build the runtime environment JSON
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Submit Ray job with full training pipeline
ray job submit --address="http://${MLP_WORKER_0_HOST}:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
        "PYTHONPATH": "/root/Megatron-LM/",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
        "NCCL_IB_TC": "160",
        "NCCL_PXN_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_NET_GDR_LEVEL": "4",
        "NCCL_IB_RETRY_CNT": "7",
        "NCCL_IB_TIMEOUT": "32",
        "NCCL_IB_QPS_PER_CONNECTION": "8",
        "NCCL_P2P_LEVEL": "NVL",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_CTAS": "4",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "^openib",
        "OMPI_MCA_routed": "direct",
        "OMPI_MCA_routed_radix": "1024",
        "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
        "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
        "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
   --save-debug-rollout-data /mnt/artifacts/gpt-oss/data.pt \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
