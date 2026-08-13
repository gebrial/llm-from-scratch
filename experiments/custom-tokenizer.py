#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datasets

dataset = datasets.load_dataset("roneneldan/TinyStories")


# In[2]:


# dataset.cleanup_cache_files()


# In[3]:


# delete any examples with none english characters
def filter_func(example):
    # remove shortest and longest 1% of stories
    if len(example["text"]) < 418 or len(example["text"]) > 2505:
        return False
    for char in example["text"]:
        if ord(char) > 127:
            return False
    return True
dataset = dataset.filter(filter_func)


# In[5]:


from tokenizers import Tokenizer
import tokenizers.decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import BPEDecoder
import tokenizers


tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = tokenizers.decoders.ByteLevel()


# In[6]:


from tokenizers.trainers import BpeTrainer


trainer = BpeTrainer(
    special_tokens=["<|endoftext|>", "\n", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=4096,
    show_progress=True,
)

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset["train"]), batch_size):
        print(f"Processing batch {i} to {i + batch_size}")
        yield dataset["train"][i:i + batch_size]["text"]

tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
)


# In[7]:


vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")


# In[8]:


print(tokenizer.decode(list(range(vocab_size, vocab_size - 100, -1)), skip_special_tokens=False))
print(tokenizer.decode(list(range(0, 100)), skip_special_tokens=False))


# In[ ]:





# In[9]:


tokenizer.get_vocab()


# In[ ]:





# In[10]:


print(dataset["validation"][0]["text"])


# In[11]:


output = tokenizer.encode(dataset["validation"][0]["text"])
print(output.tokens)


# In[14]:


print(tokenizer.decode(output.ids, skip_special_tokens=False))


# In[15]:


validation_encodings = tokenizer.encode_batch_fast(dataset["validation"]["text"])


# In[16]:


lengths = [len(x.ids) for x in validation_encodings]


# In[17]:


import numpy as np

# print stats
print("Mean:", np.mean(lengths))
print("Median:", np.median(lengths))
print("Min:", np.min(lengths))
print("Max:", np.max(lengths))
print("Std:", np.std(lengths))


# In[ ]:





# In[18]:


tokenizer.save("TinyStories_tokenizer_small_cleaned_BPE.json")


# In[19]:


from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("TinyStories_tokenizer_small_cleaned_BPE.json")


# In[20]:


tokenizer.encode("<|endoftext|>").tokens


# In[21]:


output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)


# In[22]:


tokenizer.get_vocab_size()


# In[24]:


tokenizer.get_vocab()


# In[ ]:




