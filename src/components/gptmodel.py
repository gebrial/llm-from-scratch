import torch
import torch.nn as nn

from .transformer import TransformerBlock


class GPTModel(nn.Module):
    """GPT-style decoder-only transformer.

    Uses RoPE for positional information by default (see cfg["no_pos_emb"]) and
    expects packed, masked sequences: forward() takes [token_ids, attn_mask, positions]
    rather than plain token ids, so multiple stories can share one training sequence.
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.no_pos_emb = cfg.get("no_pos_emb", False)  # https://arxiv.org/abs/2305.19466
        if not self.no_pos_emb:
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # a ModuleList rather than nn.Sequential, so forward() can pass each block its
        # own KV cache; both name parameters trf_blocks.0..., so checkpoints still load
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        if cfg.get("weight_tying", False):
            self.out_head.weight = self.tok_emb.weight
            self.out_head.bias = nn.Parameter(torch.zeros(cfg["vocab_size"]))

    def forward(self, inp, past_kv=None, use_cache=False):
        """
        Returns logits, or (logits, kv caches) with use_cache=True. past_kv is the
        list of per-layer caches from a previous call (None to start a new one);
        caching needs attn_mask=None, i.e. a single story rather than a packed sequence.
        """
        in_idx, attn_mask, positions = inp
        _, seq_len = in_idx.shape

        x = self.tok_emb(in_idx)
        if not self.no_pos_emb:
            x = x + self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = self.drop_emb(x)
        if past_kv is None:
            past_kv = [None] * len(self.trf_blocks)
        new_kv = []
        for block, kv_cache in zip(self.trf_blocks, past_kv):
            x, kv_cache = block([x, attn_mask, positions], kv_cache=kv_cache)
            new_kv.append(kv_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return (logits, new_kv) if use_cache else logits
