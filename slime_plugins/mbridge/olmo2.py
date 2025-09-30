from mbridge.core import LLMBridge, register_model


@register_model("olmo2")
class OLMo2Bridge(LLMBridge):
    """
    Bridge implementation for Qwen2 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen2 models, handling the conversion between
    Hugging Face Qwen2 format and Megatron-Core.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
        "post_self_attn_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    _NORMALIZATION_MAPPINGS = {
        "post_self_attn_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "post_mlp_layernorm.weight": ["model.layers.{layer_number}.post_feedforward_layernorm.weight"],
    }

    def _weight_name_mapping_normalization(self, name: str) -> list[str]:
        """
        Map attention weight names from MCore to Hugging Face.

        Args:
            name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._NORMALIZATION_MAPPINGS.items():
            if keyword in name:
                convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        elif "post" in mcore_weights_name:
            return self._weight_name_mapping_normalization(mcore_weights_name)
        elif "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _build_config(self):
        return self._build_base_config()
