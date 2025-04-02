#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


valid_file_loc = "../data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
with open(valid_file_loc, encoding="utf-8") as f:
    valid_text = f.read()

valid_text_split = valid_text.split("<|endoftext|>")


# In[3]:


from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("./TinyStories_tokenizer.json")


# In[4]:


get_ipython().run_cell_magic('time', '', '\nstory_tokens_counts = []\n\nfor story in valid_text_split:\n    story_tokens = tokenizer.encode(story).ids\n    story_tokens_counts.append(len(story_tokens))\n')


# In[11]:


get_ipython().run_cell_magic('time', '', '\nstory_tokens_counts = [len(tokenizer.encode(txt)) for txt in valid_text_split]\nprint(len(story_tokens_counts))\n')


# In[5]:


get_ipython().run_cell_magic('time', '', '\nstory_tokens_counts = []\n\nstory_tokens = tokenizer.encode_batch(valid_text_split)\n[story_tokens_counts.append(len(s)) for s in story_tokens]\nprint(len(story_tokens_counts))\n')


# In[6]:


get_ipython().run_cell_magic('time', '', '\nstory_tokens_counts = [len(tok) for tok in tokenizer.encode_batch(valid_text_split)]\nprint(len(story_tokens_counts))\n')


# In[7]:


import matplotlib.pyplot as plt

plt.hist(story_tokens_counts, bins=40)


# In[13]:


import numpy as np

count = np.sum(np.array(story_tokens_counts) < 128)
print("proportion of stories with less than 128 tokens: ", count / (len(story_tokens_counts)))


# In[14]:


count = np.sum(np.array(story_tokens_counts) < 256)
print("proportion of stories with less than 256 tokens: ", count / (len(story_tokens_counts)))


# In[15]:


count = np.sum(np.array(story_tokens_counts) < 512)
print("proportion of stories with less than 512 tokens: ", count / (len(story_tokens_counts)))


# In[ ]:





# count how many time each token shows up, and plot a histogram

# In[ ]:




