#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datasets

dataset = datasets.load_dataset("roneneldan/TinyStories")


# In[2]:


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()


# In[3]:


from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase, BertNormalizer, Replace

tokenizer.normalizer = normalizers.Sequence([
    BertNormalizer(),
    NFD(),
    Replace("’", "'"),
    Replace("‘", "'"),
    Replace("“", '"'),
    Replace("”", '"'),
    Replace("–", "-"),
    Replace("—", "-"),
    Replace("…", "..."),
    Replace("´", "'"),
    Replace("`", "'"),
    Replace("一", "-"),
    # Replace("🌴", ""),
    # Replace("🍌", ""),
    Replace("─", "-"),
    # Replace("€", ""),
    # Replace("™", ""),
    # Replace("œ", ""),
    # Replace("˜", ""),
    # Replace("¦", ""),
    # Replace(r'[\u4E00-\u9FFF]', ''), # Remove all CJK Unified Ideographs (and other non-ASCII)
    # Replace(r'[^\x00-\x7F]', ''), # Remove non-ASCII (Chinese, etc.)
    # Lowercase(),
    # StripAccents()
])


# In[4]:


test_string = "Hello, y'all! How are you 😁 <|endoftext|> \n? 奮 些 ä Héllø 中国 123! 巴  恩  和  艾  米  莉  兩  兒  童  在  一  個  玉  米  田  裡  度  過  了  一  整  天"
tokenizer.normalizer.normalize_str(test_string)


# In[ ]:


from tokenizers.trainers import BpeTrainer


# allowed chars chould only be utf-8 basic latin set
allowed_chars = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?\"'()[]{}<>@#$%^&*+-=~`|\\/"
    "\n\t\r ¡"
)

trainer = BpeTrainer(
    special_tokens=["<|endoftext|>", "\n", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=4096,
    min_frequency=2,
    show_progress=True,
)

def filter_text(text):
    return ''.join(c for c in text if c in allowed_chars)

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset["train"]), batch_size):
        print(f"Processing batch {i} to {i + batch_size}")
        yield [filter_text(text) for text in dataset["train"][i:i + batch_size]["text"]]

tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
)


# In[10]:


output = tokenizer.encode("Hello, y'all! How are you 😁 <|endoftext|> \n? 奮 些 ä Héllø 中国 123!")
print(output.tokens)
print(output.ids)


# In[13]:


vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")


# In[17]:


print(tokenizer.decode(list(range(vocab_size, vocab_size - 100, -1)), skip_special_tokens=False))
print(tokenizer.decode(list(range(0, 100)), skip_special_tokens=False))


# In[ ]:





# In[21]:


tokenizer.get_vocab()


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', '\n')


# In[23]:


dataset["validation"][0]


# In[24]:


output = tokenizer.encode(dataset["validation"][0]["text"])
print(output.tokens)


# In[25]:


validation_encodings = tokenizer.encode_batch_fast(dataset["validation"]["text"])


# In[26]:


lengths = [len(x.ids) for x in validation_encodings]


# In[27]:


import numpy as np

# print stats
print("Mean:", np.mean(lengths))
print("Median:", np.median(lengths))
print("Min:", np.min(lengths))
print("Max:", np.max(lengths))
print("Std:", np.std(lengths))


# In[ ]:





# In[28]:


tokenizer.save("TinyStories_tokenizer_small_cleaned.json")


# In[29]:


from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("TinyStories_tokenizer_small_cleaned.json")


# In[32]:


tokenizer.encode("<|endoftext|>").tokens


# In[33]:


output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)


# In[34]:


tokenizer.get_vocab_size()


# In[35]:


tokenizer.get_vocab()


# In[ ]:




