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





# In[ ]:


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

def create_dataloader_v3(txt, batch_size=4, max_length=256, stride=256, shuffle=True, drop_last=True, num_workers=0):
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





# In[ ]:


import torch
from torch.utils.data import IterableDataset

# implements splitting text based on endoftext token
# possible improvements:
#  sequence packing + accompanying mask (start by geting distribution of training item lengths)
#  implement sliding window chunking for text longer than max_length
class GPTDatasetPacked(IterableDataset):
    def __init__(self, txt:str, tokenizer, max_length:int, split_on="<|endoftext|>"):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.text_split = txt.split(split_on)
        # add endoftext token to the end of each text
        self.text_split = [text + " " + split_on for text in self.text_split]

    def __iter__(self):
        buffer = []
        for text in self.text_split:
            tokens = self.tokenizer.encode(text).ids
            buffer.extend(tokens)
            while len(buffer) > self.max_length:
                chunk = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                yield torch.tensor(chunk), torch.tensor(chunk[1:] + [buffer[0]])



# In[ ]:


from tokenizers import Tokenizer
from torch.utils.data import DataLoader

def create_dataloader_packed2(txt:str, batch_size=32, max_length=256, drop_last=True, num_workers=0, prefetch_factor=4):
    tokenizer = Tokenizer.from_file("./TinyStories_tokenizer.json")
    dataset = GPTDatasetPacked(txt, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    return dataloader


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# from torch.utils.data import IterableDataset

# # dataset to load files from a directory and load each line as a separate training example of tokens
# # this will allow us to implement sequence packing
# class GPTDatasetV4(IterableDataset):
#     def __init__(self, files, tokenizer, max_length:int, stride:int):
#         self.files = files
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.stride = stride

#     def __iter__(self):
#         buffer = []
#         for file in self.files:
#             with open(file, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue

#                     # convert line of numbers to list of integers
#                     try:
#                         line = list(map(int, line.split()))
#                     except ValueError:
#                         continue

#                     # add to buffer
#                     buffer.append(line)

#                     # if buffer is full, yield it
#                     if len(buffer) > self.max_length:
#                         yield buffer[:self.max_length], buffer[1:self.max_length + 1]
#                         buffer = buffer[self.stride:]



# In[ ]:


# from tokenizers import Tokenizer
# from torch.utils.data import DataLoader

# def create_dataloader_v4(files, batch_size=4, max_length=256, stride=256, drop_last=True, num_workers=0):
#     tokenizer = Tokenizer.from_file("./TinyStories_tokenizer.json")
#     dataset = GPTDatasetV4(files, tokenizer, max_length, stride)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,  # shuffle is not supported for IterableDataset
#         drop_last=drop_last,
#         num_workers=num_workers,
#         persistent_workers=True if num_workers > 0 else False,
#         pin_memory=True,
#         prefetch_factor=4 if num_workers > 0 else None
#     )

#     return dataloader

