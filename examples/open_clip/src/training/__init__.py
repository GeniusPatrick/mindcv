# https://github.com/mlfoundations/open_clip/discussions/56
# https://github.com/mlfoundations/open_clip/issues/22
# https://github.com/mlfoundations/open_clip/issues/300
# https://github.com/mlfoundations/open_clip/issues/448

# cc3m
# 2,862,387 ???
# lr=1e-3, wd=0.1, bs=128, epoch=30, model=RN50, ~ 20%
# RN50x4, ~ 22.2%

# cc12m
# 10,062,127 ???
# ~ 36%

# laion-400m
# ViT-B/32 was trained with 128 A100 (40 GB) GPUs for ~36 hours, 4600 GPU-hours.
# Batch size per GPU was 256 for a global batch size of 32768.
# reaching a top-1 ImageNet-1k zero-shot accuracy of 62.96%, comparable to OpenAI's 63.2%
# ViT-B/16 was trained with 176 A100 (40 GB) GPUS for ~61 hours, 10700 GPU-hours.
# Batch size per GPU was 192 for a global batch size of 33792.
# ViT-B/16 achieve an accuracy of 67.1%, comparable to OpenAI's 68.3%
# ViT-L/14 was trained with 400 A100 (40 GB) GPUS for ~127 hours, 50800 GPU-hours.
# Batch size per GPU was 96 for a global batch size of 38400. Grad checkpointing was enabled.
# ViT-L/14 achieve an accuracy of 72.77%, vs OpenAI's 75.5%

# LAION-2B
# ViT-B/32 was trained with 112 A100 (40 GB) GPUs.
# Batch size per GPU was 416 for a global batch size of 46592.
# reaching a top-1 ImageNet-1k zero-shot accuracy of 65.62%.
