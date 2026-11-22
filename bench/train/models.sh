# Fixed compute: active FFN width k*d = 8192, total expert width E*d = 65536 per layer.
H=2048; HEADS=16; SEQ=${SEQ:-4096}; LAYERS=${LAYERS:-16}
case $MODEL in
  dense) MODEL_ARGS=(--ffn-hidden-size 8192) ;;
  g1)  MODEL_ARGS=(--num-experts 8   --moe-router-topk 1  --moe-ffn-hidden-size 8192) ;;
  g2)  MODEL_ARGS=(--num-experts 16  --moe-router-topk 2  --moe-ffn-hidden-size 4096) ;;
  g4)  MODEL_ARGS=(--num-experts 32  --moe-router-topk 4  --moe-ffn-hidden-size 2048) ;;
  g8)  MODEL_ARGS=(--num-experts 64  --moe-router-topk 8  --moe-ffn-hidden-size 1024) ;;
  g16) MODEL_ARGS=(--num-experts 128 --moe-router-topk 16 --moe-ffn-hidden-size 512) ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac
MODEL_ARGS+=(
  --num-layers $LAYERS --hidden-size $H --num-attention-heads $HEADS
  --seq-length $SEQ --max-position-embeddings $SEQ
  --swiglu --normalization RMSNorm --position-embedding-type rope
  --disable-bias-linear --untie-embeddings-and-output-weights
)
[ "$MODEL" = dense ] || MODEL_ARGS+=(
  --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2
  --moe-router-dtype fp32 --moe-router-force-load-balancing
)
