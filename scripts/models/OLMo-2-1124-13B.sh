# This comes from LLAMA 3.1 mainly adapate to Olmo 1 Arch with the Modification of Olmo 2
MODEL_ARGS=(
   --spec "slime_plugins.models.olmo2" "get_olmo2_spec"
   --swiglu
   --num-layers 40
   --hidden-size 5120
   --ffn-hidden-size 13824
   --num-attention-heads 40
   --num-query-groups 40
   # --group-query-attention
   # --num-query-groups 8
   --max-position-embeddings 4096
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-06
   --rotary-base 500000
   --vocab-size 100352
   # --kv-channels 128
   # --use-rope-scaling
   # --rotary-scaling-factor 8.0
   --untie-embeddings-and-output-weights

   # Args from the slime patch
   --qk-layernorm
   --post-self-attn-layernorm
   --post-mlp-layernorm
   --qk-layernorm-unshare
)
