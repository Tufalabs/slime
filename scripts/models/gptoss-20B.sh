# GPT-OSS 20B Model Configuration
# Based on OpenAI's GPT-OSS architecture with 24 layers and 32 experts

# gptoss-20B
MODEL_ARGS=(
    --spec "slime_plugins.models.gptoss" "get_gptoss_spec"

    # Basic architecture
    --num-layers 24
    --hidden-size 2880
    --num-attention-heads 64
    --num-query-groups 8
    --group-query-attention
    --kv-channels 64
    --ffn-hidden-size 2880

    # Normalization and bias
    --normalization RMSNorm
    # Note: GPT-OSS has biases, so we DON'T use --disable-bias-linear
    --untie-embeddings-and-output-weights
    --vocab-size 201088

    # Position embeddings - RoPE with YARN-like scaling
    # Note: Megatron-LM's arguments.py doesn't include 'yarn' as a valid choice
    # Use standard RoPE with appropriate base and scaling for long context
    --position-embedding-type rope
    --rotary-base 150000
    --rotary-scaling-factor 32.0
    --mscale 1.0
    --mscale-all-dim 1.0
    --rotary-percent 1.0
    --seq-length 131072

    # Sliding window attention
    --window-size "128,0"
    --window-attn-skip-freq 2

    # Activation function
    --quick-geglu
    --activation-func-clamp-value 7.0
    --glu-linear-offset 1.0

    # Fusion flags
    # GPT-OSS has biases (add_bias_linear=True by default)
    # bias_dropout_fusion is NOT supported in TEGroupedMLP when add_bias_linear=True
    --no-bias-dropout-fusion

    # Learnable softmax
    --softmax-type learnable

    # MoE configuration
    --num-experts 32
    --moe-router-topk 4
    --moe-ffn-hidden-size 2880
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-load-balancing-type none

    # Dropouts (set to 0 for conversion, can be overridden during training)
    --hidden-dropout 0.0
    --attention-dropout 0.0
)
