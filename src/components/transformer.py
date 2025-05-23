#!/usr/bin/env python
# coding: utf-8

# In[1]:


from .attention import MultiHeadAttention
import torch.nn as nn
from .feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from .attention import MultiHeadAttention_v2
import torch.nn as nn
from .feedforward import FeedForward

class TransformerBlock_v2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention_v2(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# In[ ]:


from .attention import MultiHeadAttention_v3
import torch.nn as nn
from .feedforward import FeedForward

class TransformerBlock_v3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention_v3(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, inp):
        x, attn_mask = inp

        shortcut = x
        x = self.norm1(x)
        x = self.att([x, attn_mask])
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return [x, attn_mask]


# In[ ]:


from .attention import MultiHeadAttention_v3
import torch.nn as nn
from .feedforward import FeedForwardGeGLU

class TransformerBlock_GeGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention_v3(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForwardGeGLU(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, inp):
        x, attn_mask = inp

        shortcut = x
        x = self.norm1(x)
        x = self.att([x, attn_mask])
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return [x, attn_mask]


# In[ ]:


from .attention import MultiHeadAttention_RoPE
import torch.nn as nn
from .feedforward import FeedForward

class TransformerBlock_RoPE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention_RoPE(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, inp):
        x, attn_mask, positions = inp

        shortcut = x
        x = self.norm1(x)
        x = self.att([x, attn_mask, positions])
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return [x, attn_mask, positions]


# In[ ]:





# In[ ]:




