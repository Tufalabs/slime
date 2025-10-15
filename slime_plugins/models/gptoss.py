from typing import Optional

import transformer_engine as te  # pylint: disable=unused-import
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def get_gptoss_layer_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
):
    backend = TESpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=False,
        use_te_op_fuser=False,
        use_te_activation_func=False,
    )

    spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_layer_norm_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layer_norm(),
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
            },
        ),
    )

    return spec


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

    # Hack: Add YARN parameters to config if they don't exist
    if not hasattr(config, "position_embedding_type"):
        config.position_embedding_type = "yarn"
    if not hasattr(config, "yarn_rotary_scaling_factor"):
        config.yarn_rotary_scaling_factor = 32.0
    if not hasattr(config, "yarn_original_max_position_embeddings"):
        config.yarn_original_max_position_embeddings = 4096
    if not hasattr(config, "yarn_beta_fast"):
        config.yarn_beta_fast = 32.0
    if not hasattr(config, "yarn_beta_slow"):
        config.yarn_beta_slow = 1.0
    if not hasattr(config, "yarn_correction_range_round_to_int"):
        config.yarn_correction_range_round_to_int = False
    if not hasattr(config, "yarn_mscale"):
        config.yarn_mscale = 1.0
    if not hasattr(config, "yarn_mscale_all_dim"):
        config.yarn_mscale_all_dim = 1.0

    transformer_layer_spec = get_gptoss_layer_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
    )
    return transformer_layer_spec
