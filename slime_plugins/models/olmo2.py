from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

def get_olmo2_spec(args):
    print("Im entering here")
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        qk_layernorm=args.qk_layernorm,
        post_self_attn_layernorm=args.post_self_attn_layernorm,
        post_mlp_layernorm=args.post_mlp_layernorm,
    )

    transformer_layer_spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
    transformer_layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear


    return transformer_layer_spec
