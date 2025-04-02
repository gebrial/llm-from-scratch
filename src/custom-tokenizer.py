#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())


# In[2]:


from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()


# In[3]:


get_ipython().run_cell_magic('time', '', '\nfrom tokenizers.trainers import BpeTrainer\n\ndata_folder = "../data/TinyStories/"\nfiles = [\n    data_folder + "TinyStoriesV2-GPT4-train.txt",\n    # data_folder + "TinyStoriesV2-GPT4-valid.txt",\n]\n\ntrainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])\ntokenizer.train(files=files, trainer=trainer)\n')


# In[4]:


output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]


# In[5]:


tokenizer.get_vocab_size()


# In[ ]:





# In[16]:


get_ipython().run_cell_magic('time', '', '\nvalid_file = "../data/TinyStories/TinyStoriesV2-GPT4-valid.txt"\nwith open(valid_file, "r", encoding="utf-8") as f:\n    valid_text = f.read()\n')


# In[17]:


valid_text[:1000]


# In[8]:


tokenizer.add_special_tokens(["<|endoftext|>", "\n"])


# In[18]:


output = tokenizer.encode(valid_text[:1000])
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]


# In[ ]:





# In[10]:


tokenizer.save("TinyStories_tokenizer.json")


# In[11]:


tokenizer.from_file("TinyStories_tokenizer.json")


# In[12]:


output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]


# In[ ]:





# In[ ]:




