"""
GPT-OSS layer specification for SLIME.

GPT-OSS uses standard Megatron-Core GPT layer specs with specific configuration:
- MoE with grouped GEMM
- No QK layernorm
- No Multi-Latent Attention (MLA)
- Standard transformer engine implementation
"""

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


def get_gptoss_spec(args, config, vp_stage):
    """
    Get the layer specification for GPT-OSS models.

    GPT-OSS uses standard GPT layers but with specific architectural features
    configured via the TransformerConfig and command-line arguments:
    - YARN RoPE for position embeddings
    - Sliding window attention with alternating pattern
    - Learnable softmax with per-layer offsets
    - Quick GELU activation with clamping
    - MoE with top-4 routing

    Args:
        args: Command-line arguments containing model configuration
        config: TransformerConfig object with GPT-OSS specific settings
        vp_stage: Virtual pipeline stage (for pipeline parallelism)

    Returns:
        TransformerLayerSpec: Layer specification for GPT-OSS
    """
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=False,  # GPT-OSS does not use QK normalization
        multi_latent_attention=False,  # GPT-OSS uses standard attention
        moe_use_legacy_grouped_gemm=False,
    )
    return transformer_layer_spec
