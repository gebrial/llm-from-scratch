#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # Linear projections for the gating and value branches
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        # Project input into 2 parts: one for value, one for gate
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)  # Apply GELU to the gate and multiply

class FeedForwardGeGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(cfg["emb_dim"], cfg["emb_dim"] * 4),  # GeGLU activation
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])  # Output projection
        )

    def forward(self, x):
        return self.net(x)

