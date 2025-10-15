#!/bin/bash

# GPT-OSS 120B Training Script

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
export PYTHONBUFFERED=1

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
   --save $BASE_DIR/gpt-oss-20b_slime
   --save-interval 1
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

   --global-batch-size 8
   --balance-data
   --rollout-stop-token-ids 151329 151336 151338  # Adjust for GPT-OSS tokenizer
)

# Evaluation configuration
EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data aime $BASE_DIR/rl_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

# Performance and parallelism configuration
# Adjust these based on your hardware setup
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

# GRPO/GSPO configuration
GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

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
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
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

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}\",
    \"GLOO_SOCKET_IFNAME\": \"${MLP_SOCKET_IFNAME}\",
    \"TP_SOCKET_IFNAME\": \"${MLP_SOCKET_IFNAME}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_P2P_LEVEL\": \"NVL\",
    \"NCCL_CUMEM_ENABLE\": \"0\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_NET_GDR_LEVEL\": \"4\",
    \"NCCL_PXN_DISABLE\": \"0\",
    \"NCCL_MIN_CTAS\": \"4\",
    \"NVTE_BWD_LAYERNORM_SM_MARGIN\": \"20\",
    \"TORCH_NCCL_AVOID_RECORD_STREAMS\": \"1\"
  }
}"

# Submit Ray job with full training pipeline
ray job submit --address="http://${MLP_WORKER_0_HOST}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
