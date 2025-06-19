export RWKV_JIT_ON=0

# DS_BUCKET_MB=200 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)

CUDA_VISIBLE_DEVICES=4 python train_rwkv.py \
    --strategy deepspeed_stage_2 \
    --precision fp32 \
    --load_model "0" \
    --train_stage 2 \
    --accelerator gpu \
    --batch_size 256 \
    # --ds_bucket_mb $DS_BUCKET_MB