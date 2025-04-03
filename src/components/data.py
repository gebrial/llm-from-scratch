#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[30]:


import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"}) 
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# In[31]:


import tiktoken
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# In[ ]:





# In[ ]:





# In[9]:


import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV2(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt).ids
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# In[8]:


from tokenizers import Tokenizer

def create_dataloader_v2(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = Tokenizer.from_file("./TinyStories_tokenizer.json")
    dataset = GPTDatasetV2(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# In[ ]:





# In[ ]:





# In[2]:


import torch
from torch.utils.data import Dataset, DataLoader

# implements splitting text based on endoftext token
# possible improvements:
#  sequence packing + accompanying mask (start by geting distribution of training item lengths)
#  implement sliding window chunking for text longer than max_length
class GPTDatasetV3(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, split_on="<|endoftext|>"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.endoftext_id = tokenizer.encode(split_on).ids[0]

        self.text_split = txt.split(split_on)


    def _pad_tokens_torch(self, tokens, max_length, pad_token_id):
        tensor = torch.full((max_length,), pad_token_id, dtype=torch.long)
        tokens = torch.tensor(tokens[:max_length])
        tensor[:len(tokens)] = tokens
        return tensor

    def __len__(self):
        return len(self.text_split)

    def __getitem__(self, idx):
        # max_length = min(self.max_length, self.max_tokens_length)
        text = self.text_split[idx]
        tokens = self.tokenizer.encode(text).ids
        tokens_padded = self._pad_tokens_torch(tokens, self.max_length+1, self.endoftext_id)
        return tokens_padded[:self.max_length], tokens_padded[1:self.max_length + 1]


# In[ ]:


from tokenizers import Tokenizer

def create_dataloader_v3(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = Tokenizer.from_file("./TinyStories_tokenizer.json")
    dataset = GPTDatasetV3(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None
    )

    return dataloader


# In[ ]:




