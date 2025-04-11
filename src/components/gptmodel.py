#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
from .transformer import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.no_pos_emb = cfg.get("no_pos_emb", False) # https://arxiv.org/abs/2305.19466


        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        self.weight_tying = cfg.get("weight_tying", False)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        if self.weight_tying:
            self.out_head.weight = self.tok_emb.weight
            self.out_head.bias = nn.Parameter(torch.zeros(cfg["vocab_size"]))


        self.seq_layers = nn.Sequential(
            self.drop_emb,
            self.trf_blocks,
            self.final_norm,
            self.out_head
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        if not self.no_pos_emb:
            pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
            )
            x = x + pos_embeds

        logits = self.seq_layers(x)
        return logits

        # x = self.drop_emb(x)
        # x = self.trf_blocks(x)
        # x = self.final_norm(x)
        # logits = self.out_head(x)
        # return logits


# In[ ]:





# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
from .transformer import TransformerBlock_v2

class GPTModel_v2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.no_pos_emb = cfg.get("no_pos_emb", False) # https://arxiv.org/abs/2305.19466


        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock_v2(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        self.weight_tying = cfg.get("weight_tying", False)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        if self.weight_tying:
            self.out_head.weight = self.tok_emb.weight
            self.out_head.bias = nn.Parameter(torch.zeros(cfg["vocab_size"]))


        # self.seq_layers = nn.Sequential(
        #     self.drop_emb,
        #     self.trf_blocks,
        #     self.final_norm,
        #     self.out_head
        # )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        if not self.no_pos_emb:
            pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
            )
            x = x + pos_embeds

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[ ]:


import torch
import torch.nn as nn
from .transformer import TransformerBlock_v3

class GPTModel_v3(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.no_pos_emb = cfg.get("no_pos_emb", False) # https://arxiv.org/abs/2305.19466


        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock_v3(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        self.weight_tying = cfg.get("weight_tying", False)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        if self.weight_tying:
            self.out_head.weight = self.tok_emb.weight
            self.out_head.bias = nn.Parameter(torch.zeros(cfg["vocab_size"]))

    def forward(self, inp):
        in_idx, attn_mask = inp
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        if not self.no_pos_emb:
            pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
            )
            x = x + pos_embeds

        x = self.drop_emb(x)
        x, _ = self.trf_blocks([x, attn_mask])
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[ ]:


import torch
import torch.nn as nn
from .transformer import TransformerBlock_GeGLU

class GPTModel_GeGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.no_pos_emb = cfg.get("no_pos_emb", False) # https://arxiv.org/abs/2305.19466


        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock_GeGLU(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        self.weight_tying = cfg.get("weight_tying", False)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        if self.weight_tying:
            self.out_head.weight = self.tok_emb.weight
            self.out_head.bias = nn.Parameter(torch.zeros(cfg["vocab_size"]))

    def forward(self, inp):
        in_idx, attn_mask = inp
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        if not self.no_pos_emb:
            pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
            )
            x = x + pos_embeds

        x = self.drop_emb(x)
        x, _ = self.trf_blocks([x, attn_mask])
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[ ]:


import torch
import torch.nn as nn
from .transformer import TransformerBlock_RoPE

class GPTModel_RoPE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.no_pos_emb = cfg.get("no_pos_emb", False) # https://arxiv.org/abs/2305.19466


        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock_RoPE(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        self.weight_tying = cfg.get("weight_tying", False)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        if self.weight_tying:
            self.out_head.weight = self.tok_emb.weight
            self.out_head.bias = nn.Parameter(torch.zeros(cfg["vocab_size"]))

    def forward(self, inp):
        in_idx, attn_mask, positions = inp
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        if not self.no_pos_emb:
            pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
            )
            x = x + pos_embeds

        x = self.drop_emb(x)
        x, attn_mask, positions = self.trf_blocks([x, attn_mask, positions])
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# In[ ]:




