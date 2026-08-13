# src/components

The model implementation, imported by the pipeline scripts in `../scripts/`.

| File | Contents |
|---|---|
| `attention.py` | `MultiHeadAttention`: causal multi-head attention with RoPE, built on `scaled_dot_product_attention` |
| `feedforward.py` | `FeedForward`: the transformer block's MLP (GELU, 4x expansion) |
| `transformer.py` | `TransformerBlock`: attention and feedforward with pre-norm and residual connections |
| `gptmodel.py` | `GPTModel`: token embedding, stack of transformer blocks, output head |
| `generatetext.py` | `top_p_sampling`, `beam_search`, `generate_text`: the inference-time decoding functions |

Everything here operates on packed, masked sequences. `GPTModel.forward`
takes `[token_ids, attn_mask, positions]` instead of plain token ids, so
multiple stories can share one training sequence without attending into
each other (see `scripts/prepare_dataset.py` for how the mask and positions
get built).

See the top-level [README](../README.md) for how this fits into the full
pipeline.
