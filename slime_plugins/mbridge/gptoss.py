import json
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List

import torch
from safetensors import safe_open

from mbridge.core import LLMBridge, register_model

logger = logging.getLogger(__name__)


class SafeTensorIMXFP4:
    def __init__(self, hf_dir: str):
        index_file = os.path.join(hf_dir, "model.safetensors.index.json")

        self.index: Dict[str, str] = {}
        self.origin_index = {}
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                origin_index = json.load(f)
                self.index = origin_index["weight_map"]
                self.origin_index = origin_index
        else:
            raise ValueError("No model.safetensors.index.json foun in the {hf_dir}")

        self.hf_dir = hf_dir

    def preprocess_index_for_mxfp4(self, index: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        # In the hf mapping to safetensors, block and scales might point to different files.
        # mbridge will just pass the param without the suffix block and scale. For
        # this case we will return a dict from the index[name] otherwise just a a path.
        # I.e idex[name] -> {name: safe_tensor_path} or {*_scale: safe_tensor_path, *_block: safee_tensor_path}

        new_index: Dict[str, Dict[str, str]] = defaultdict(dict)

        for key, value in index.items():
            new_key = key

            if key.endswith("blocks"):
                new_key = key[0 : -(len("blocks") + 1)]

            if key.endswith("scales"):
                new_key = key[0 : -(len("scales") + 1)]

            new_index[new_key][key] = value

        return new_index

    def load_one_hf_weight(self, hf_weight_name: str) -> torch.Tensor:
        return self.load_some_hf_weight([hf_weight_name])[hf_weight_name]

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> Dict[str, torch.Tensor]:
        index = self.index
        hf_dir = self.hf_dir

        loaded_mxfp4_data: Dict[str, torch.Tensor] = {}
        loaded_bf16_data: Dict[str, torch.Tensor] = {}

        index = self.preprocess_index_for_mxfp4(index)
        file_to_weight_map: Dict[str, list[str]] = defaultdict(list)

        for name in hf_weight_names:
            filenames_dict = index[name]
            for param_name, filename in filenames_dict.items():
                file_to_weight_map[filename].append(param_name)

        for filename, weight_names in file_to_weight_map.items():
            safetensor_file = os.path.join(hf_dir, filename)
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for name in weight_names:
                    if name.endswith("blocks") or name.endswith("scales"):
                        loaded_mxfp4_data[name] = f.get_tensor(name)
                    else:
                        loaded_bf16_data[name] = f.get_tensor(name)

        # Convert MXFP4 weights to BF16
        for k, v in loaded_mxfp4_data.items():
            if k.endswith("scales"):
                continue  # process scales in the iteration of blocks
            blocks = v
            scales = loaded_mxfp4_data[k.replace("blocks", "scales")].to(torch.int32) - 127
            new_key = k.replace(".blocks", "").replace("_blocks", "")
            loaded_bf16_data[new_key] = self._dequantize_mxfp4(blocks, scales)
            logging.debug(f"Successfully dequantized {new_key}")

        return loaded_bf16_data

    def _dequantize_mxfp4(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
        FP4_VALUES = [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


@register_model("gpt_oss")
class GPTOSSBridge(LLMBridge):

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_proj.bias": ["model.layers.{layer_number}.self_attn.o_proj.bias"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "self_attention.core_attention.softmax_offset": ["model.layers.{layer_number}.self_attn.sinks"],
        "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
    }

    _EXPERTS_MAPPING = {
        "mlp.experts.linear_fc1.weight": ["model.layers.{layer_number}.mlp.experts.gate_up_proj"],
        "mlp.experts.linear_fc1.bias": ["model.layers.{layer_number}.mlp.experts.gate_up_proj_bias"],
        "mlp.experts.linear_fc2.weight": ["model.layers.{layer_number}.mlp.experts.down_proj"],
        "mlp.experts.linear_fc2.bias": ["model.layers.{layer_number}.mlp.experts.down_proj_bias"],
    }

    def _weight_name_mapping_experts(self, name: str):
        layer_number = name.split(".")[2]
        convert_names: List[str] = []

        for keyword, mapping_names in self._EXPERTS_MAPPING.items():
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

        elif "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name and "experts" not in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        elif "experts" in mcore_weights_name:
            return self._weight_name_mapping_experts(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _build_config(self):
        pass

    def _extract_expert(self, param_name: str, hf_weight: torch.Tensor) -> List[torch.Tensor]:
        expert_str = param_name.split(".")[-1]

        if "weight" in expert_str:
            expert_num = int(expert_str[len("weight") :])
        elif "bias":
            expert_num = int(expert_str[len("bias") :])
        else:
            raise NotImplementedError(f"Unsupported parameter name: {param_name}")

        if "linear_fc1" in param_name:
            expert_tensor = hf_weight[expert_num]
            gate = expert_tensor[::2, ...]
            up = expert_tensor[1::2, ...]
            return [gate, up]

        elif "linear_fc2" in param_name:
            expert_tensor = hf_weight[expert_num]
            return [expert_tensor]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {param_name}")

    def load_weights(
        self,
        models: list[torch.nn.Module],
        weights_path: str,
        memory_efficient: bool = False,
    ) -> None:
        """
        Load weights from a Hugging Face model into a Megatron-Core model.

        Args:
            models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)
            weights_path: Path to the weights file or Hugging Face model identifier
        """
        safetensor_io = SafeTensorIMXFP4(weights_path)

        for i, model in enumerate(models):
            # map local weight names to global weight names
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            # map local weight names to huggingface weight names
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }
            # only tp_rank0/etp_rank0 load from disk, others load from tp_rank0/etp_rank0
            to_load_from_disk = []
            for local_name, hf_names in local_to_hf_map.items():
                if ".mlp.experts.linear_fc" in local_name:
                    if self.mpu.etp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                    else:
                        # special case for lm_head.weight
                        # if make value model, every tp rank will load lm_head.weight
                        if "lm_head.weight" in hf_names:
                            to_load_from_disk.extend(hf_names)

            # load huggingface weights
            if not memory_efficient:
                hf_weights_map = safetensor_io.load_some_hf_weight(to_load_from_disk)

            # import mcore weights
            for local_name, hf_names in local_to_hf_map.items():
                print("Loading", local_name)
                param = model.state_dict()[local_name]
                # hf format to mcore format
                if set(to_load_from_disk) & set(hf_names):
                    if not memory_efficient:
                        hf_weights = [hf_weights_map[x] for x in hf_names]
                    else:
                        hf_weights = [safetensor_io.load_one_hf_weight(x) for x in hf_names]

                    # HACK: Handle expert format from HF GPT OSS.
                    # HF Ships the experts as a single matrix, and mcore
                    # separates this.
                    if ".mlp.experts.linear_fc" in local_name:
                        assert len(hf_weights) == 1, "Check the HF GPT OSS"
                        # I get the number from here and then I'll slice properly
                        # and transform it either to normal multiple hf weights
                        # using the interleave in case of fc1, or just slice it and
                        # get the normal fc2, so that then it goes to _weight_to_mcore_format.
                        hf_weights = self._extract_expert(local_name, hf_weights[0])

                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None
                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (mcore_weight is None or mcore_weight.shape[0] != 1):
                        # skip lm_head.weight when the model is a value model
                        continue

                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in local_name:
                    # split mcore weights across etp
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    # split mcore weights across tp
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.tp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                # load
                param.copy_(param_to_load)
