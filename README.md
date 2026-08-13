# llm-from-scratch

A GPT-style language model built from scratch in PyTorch, following Sebastian
Raschka's *Build a Large Language Model (From Scratch)*. From there it grew
into a full pipeline (custom tokenizer, sequence packing, RoPE, PyTorch
Lightning training) for pretraining a ~120M parameter model on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
to generate short children's stories.

## Project layout

```
src/components/   the model itself: attention, transformer block, feedforward,
                  GPTModel, and text generation (top-p sampling, beam search, greedy)
scripts/          CLI pipeline to reproduce a trained model end to end
experiments/      ~70 notebooks from earlier iterations of the architecture and
                  training pipeline, kept for history and not maintained
data/TinyStories/ git submodule for the TinyStories dataset. scripts/ don't
                  actually need it, since they pull TinyStories from the
                  Hugging Face Hub directly; only a few of the older
                  notebooks in experiments/ use the local copy
```

Current architecture, in `src/components/`:
- multi-head self-attention with [RoPE](https://arxiv.org/abs/2305.19466)
  positional encoding, built on `torch.nn.functional.scaled_dot_product_attention`
- packed training sequences, so multiple short stories share a training
  sequence instead of each one being padded out on its own. An attention
  mask keeps stories from attending into each other
- a custom byte-level BPE tokenizer trained on TinyStories, with a much
  smaller vocabulary than GPT-2's since the text itself is simple
- optional weight tying between the token embedding and output head

## Setup

```
pip install -r requirements.txt
```

## Reproducing a trained model

The original checkpoint is gone (it was never committed, only the training
code was). To train a new one:

```
# 1. train a tokenizer on TinyStories
python scripts/prepare_tokenizer.py --output data/tokenizer.json

# 2. tokenize and pack TinyStories into fixed-length training sequences
python scripts/prepare_dataset.py --tokenizer data/tokenizer.json --output data/packed_dataset

# 3. train (see --help for model size, batch size, epochs, etc.)
python scripts/train.py --dataset data/packed_dataset --tokenizer data/tokenizer.json \
    --checkpoint-dir checkpoints/run1

# 4. generate a story from a checkpoint
python scripts/generate.py --checkpoint checkpoints/run1/<checkpoint>.ckpt \
    --tokenizer data/tokenizer.json --prompt "One day a girl walked into"
```

Training (step 3) wants a GPU. The default config is 16 layers, 768 dim, 24
heads, 512 tokens of context, about 120M parameters, and trains in bf16 with
`torch.compile` on by default.

`generate.py` defaults to top-p (nucleus) sampling. In earlier experiments it
gave noticeably more varied completions than beam search, which tended to
only differ in the last few words of a story.

## History

`experiments/` has the notebooks this project went through on the way here:
the book's reference implementation first, then gradient accumulation,
OneCycleLR, weight tying, Flash Attention, sequence packing, RoPE, a custom
small-vocab tokenizer, and top-p sampling last. None of it still runs as-is
(a lot of it points at local checkpoints and datasets that don't exist
anymore), but it's there if you want to see how the current pipeline came
together.
