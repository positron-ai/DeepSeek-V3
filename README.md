# Reference DeepSeek-V3

Positron-specific reference implementation of DeepSeek-V3.  The
original README is in README_DEEPSEEK.md.

The main goal of this repository is to execute the DeepSeek-V3 model
while instrumenting it to save intermediates, so that we can validate
the Positron implementation.

## Step 1: Convert the weights

The HuggingFace weights are in a format that this repo's code can't
directly load.  Ergo, needs must convert:

```
# On a machine with some GPUs
cd inference
python3 convert.py \
  --hf-ckpt-path /opt/positron/weights/huggingface/deepseek-ai/DeepSeek-V3/ \
  --save-path /opt/positron/weights/huggingface/positron-ai/DeepSeek-V3-pt-converted-4-layers/ \
  --n-experts 256 \
  --n-layers 4 \
  --model-parallel 1
```

I believe DeepSeek's original purpose for the conversion script was to
create sharded weight sets to be able to run the model distributed
with `torchrun`, but as long as we're willing to limit the number of
layers we don't need to do that.