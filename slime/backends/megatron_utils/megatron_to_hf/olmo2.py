import re
import torch


def convert_olmo2_to_hf(args, name, param):
    """
    Convert OLMo2 model weights from Megatron-Core format to HuggingFace format.

    This function performs the INVERSE mapping of OLMo2Bridge._weight_name_mapping_mcore_to_hf
    Based on the mappings defined in /root/slime/slime_plugins/mbridge/olmo2.py
    """

    # Direct mappings (inverse of OLMo2Bridge._DIRECT_MAPPING)
    # MCore -> HF
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    # Calculate head dimensions for QKV splitting
    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # Pattern for decoder layers
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # ATTENTION MAPPINGS (inverse of OLMo2Bridge._ATTENTION_MAPPING)
        if rest == "self_attention.linear_proj.weight":
            # Maps to single o_proj
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.weight":
            # Splits into q_proj, k_proj, v_proj
            # print(f"DEBUG: Original param shape: {param.shape}")
            # print(f"DEBUG: num_query_groups: {args.num_query_groups}")
            # print(f"DEBUG: num_attention_heads: {args.num_attention_heads}")
            # print(f"DEBUG: head_dim: {head_dim}")
            # print(f"DEBUG: hidden_size: {args.hidden_size}")
            # print(f"DEBUG: value_num_per_group: {value_num_per_group}")

            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            # print(f"DEBUG: After reshape param shape: {param.shape}")

            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            # print(f"DEBUG: q_param shape before flatten: {q_param.shape}")
            # print(f"DEBUG: k_param shape before flatten: {k_param.shape}")
            # print(f"DEBUG: v_param shape before flatten: {v_param.shape}")
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]

        elif rest == "self_attention.q_layernorm.weight":
            # Maps to q_norm
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]

        elif rest == "self_attention.k_layernorm.weight":
            # Maps to k_norm
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

        # MLP MAPPINGS (inverse of OLMo2Bridge._MLP_MAPPING)
        elif rest == "mlp.linear_fc1.weight":
            # Splits into gate_proj and up_proj
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]

        elif rest == "mlp.linear_fc2.weight":
            # Maps to down_proj
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

        # NORMALIZATION MAPPINGS (inverse of OLMo2Bridge._NORMALIZATION_MAPPINGS)
        # These are specific to OLMo2 and handled specially in the bridge
        elif rest == "post_self_attn_layernorm.weight":
            # Maps to post_attention_layernorm
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        elif rest == "post_mlp_layernorm.weight":
            # Maps to post_feedforward_layernorm
            return [(f"model.layers.{layer_idx}.post_feedforward_layernorm.weight", param)]

        # Note: OLMo2 doesn't have traditional input_layernorm or post_attention_layernorm
        # in the same sense as other models. The normalization is handled differently.

    raise ValueError(f"Unknown parameter name for OLMo2: {name}")
