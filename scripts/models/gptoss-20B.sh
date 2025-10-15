# GPT-OSS 20B Model Configuration

MODEL_ARGS=(
    --spec "slime_plugins.models.gptoss" "get_gptoss_spec"

    # Basic architecture
    --num-layers 24
    --hidden-size 2880
    --num-attention-heads 64
    --num-query-groups 8
    --group-query-attention
    --ffn-hidden-size 2880
    --kv-channels 64
    --normalization "RMSNorm"
    --untie-embeddings-and-output-weights
    --vocab-size 201088
    --hidden-dropout 0.0
    --attention-dropout 0.0

    # Position embeddings (Yarn params are hooked into the config object)
    --position-embedding-type "rope"
    --rotary-base 150000

    # MoE configuration
    --moe-router-topk 4
    --num-experts 32
    --moe-grouped-gemm
    --moe-token-dispatcher-type "alltoall"
    --moe-permute-fusion
    --moe-ffn-hidden-size 2880
    --moe-router-load-balancing-type "none"
    --seq-length 131072
    --window-size "128,0"
    --softmax-type "learnable"
    --quick-geglu
    --glu-linear-offset 1.0
    --window-attn-skip-freq 2
    --activation-func-clamp-value 7.0

    --no-bias-dropout-fusion
)
