#!/usr/bin/env python
# coding: utf-8

# This script contains some improvement from train.ipynb by default:
# * much few layers, heads, and embedding dimension to reduce the model size
# * dataloader v2 which uses a custom tokenizer (again to reduce model size)
# * no positional embeddings (to reduce model complexity)
# * weight tying (to reduce model size)
# 
# 
# We implemented a few things here first and not before:
# * validation losses
# * increased the model size to be just below 30M parameters
# * reduced the amount of data trained with to keep the training (wall) time consistent
# * made graph more informative

# This script contains a couple improvements from train2.ipynb:
# * gradient accumulation is enabled
# * the dataloader chunks from the start of an example up to the max_length or the endoftext token

# This contains some changes from train3.ipynb:
# * an accuracy metric has been implemented
# * one cycle learning rate schedule is being used
# * weight tying is disabled

# This contains some improvements from train4.ipynb: just that the attention module used uses pytorchs implementation for sdpa. This also uses a text generation function to display the capabilities of the trained model.

# In[1]:


from components.gptmodel import GPTModel_v2


# In[2]:


import lightning as L


# In[ ]:





# In[3]:


import socket


# In[4]:


socket.gethostname()


# In[5]:


GPT_CONFIG_30M = {
    "vocab_size": 30002,
    "context_length": 256,
    "emb_dim": 512,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "weight_tying": True,
    "no_pos_emb": True
}


# In[6]:


GPT_CONFIG_60M = {
    "vocab_size": 30002,
    "context_length": 512,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 8,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "weight_tying": False,
    "no_pos_emb": True
}


# In[7]:


import socket
hostname = socket.gethostname().lower()
if "laptop" in hostname:
    GPT_CONFIG = GPT_CONFIG_30M
else:
    GPT_CONFIG = GPT_CONFIG_60M


# In[8]:


import torch.nn as nn
import torch

torch.set_float32_matmul_precision('medium')


# In[ ]:





# In[9]:


from components.data import create_dataloader_v3


# In[10]:


trainer_config = {
    "dataset_scale": 300,
    "batch_size": 32 if "laptop" in hostname else 32,
    "epochs": 1
}
trainer_config["grad_batches"] = 256 // trainer_config["batch_size"]


# In[11]:


def create_dataloader(text, num_workers, train=True):
    return create_dataloader_v3(
        text,
        batch_size=trainer_config["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        drop_last=train,
        shuffle=train,
        num_workers=num_workers
    )

workers = 11 if "laptop" in hostname else 23


# In[12]:


get_ipython().run_cell_magic('time', '', '\ntrain_file = "../data/TinyStories/TinyStoriesV2-GPT4-train.txt"\nwith open(train_file, "r", encoding="utf-8") as f:\n    train_text = f.read()\n\ntrain_len = len(train_text)\ntrain_text = train_text[:train_len // trainer_config["dataset_scale"]]\ntrain_loader = create_dataloader(train_text, workers)\n')


# In[13]:


val_file = "../data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
with open(val_file, "r", encoding="utf-8") as f:
    val_text = f.read()

val_loader = create_dataloader(val_text[:len(train_text)//10], workers, train=False)


# In[ ]:





# In[14]:


from torch.optim.lr_scheduler import OneCycleLR

class LitGPTModel(L.LightningModule):
    def __init__(self, GPTModel, total_steps, max_lr=1e-3):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.model = GPTModel

        self.train_accuracy = []
        self.val_accuracy = []
        self.train_losses = []
        self.val_losses = []
        self.val_steps = []
        self.learning_rates = []
        self.batch_step = 0

    def _accuracy(self, output, expected):
        total_matching = (torch.argmax(output, dim=-1) == expected).sum().item()
        total_numel = expected.numel()
        return total_matching / total_numel

    def training_step(self, batch, batch_idx):
        self.batch_step += 1
        x, y = batch
        logits = self.model(x)

        accuracy = self._accuracy(logits, y)
        self.log("accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.train_accuracy.append(accuracy)

        loss = self.loss(logits, y)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_losses.append(loss.item())

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.learning_rates.append(current_lr)

        return loss

    def validation_step(self, batch, batch_idx):
        self.val_steps.append(self.batch_step)
        x, y = batch

        logits = self.model(x)

        accuracy = self._accuracy(logits, y)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.val_accuracy.append(accuracy)

        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.val_losses.append(loss.item())

        return loss

    def loss(self, output, expected):
        loss = nn.functional.cross_entropy(
            output.flatten(0, 1), expected.flatten()
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.max_lr, weight_decay=0.1
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "loss"
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }


# In[ ]:





# In[15]:


model = GPTModel_v2(GPT_CONFIG)
litmodel = LitGPTModel(model, 1 + trainer_config["epochs"]*len(train_loader)//trainer_config["grad_batches"])


# In[16]:


get_ipython().run_cell_magic('time', '', '\ntrainer = L.Trainer(\n    max_epochs=trainer_config["epochs"],\n    enable_progress_bar=True,\n    accumulate_grad_batches=trainer_config["grad_batches"]\n)\ntrainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)\n')


# In[17]:


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(litmodel.train_losses, label="Training Loss", color="blue")
ax1.scatter(litmodel.val_steps, litmodel.val_losses, label="Validation Loss", color="red")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Loss")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(litmodel.learning_rates, label="Learning Rate", color="green")
ax2.set_ylabel("Learning Rate")
ax2.legend(loc="upper right")

plt.title("Training/Validation Loss and Learning Rate")
plt.tight_layout()
plt.grid(True)
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
plt.plot(litmodel.train_accuracy, color="blue")
plt.scatter(litmodel.val_steps, litmodel.val_accuracy, color="red")
plt.xlabel("Training Step")
plt.ylabel("Accuracy")
plt.grid(True)
plt.ylim(0, 1)
plt.show()


# In[ ]:





# In[ ]:





# In[19]:


from tokenizers import Tokenizer


# In[20]:


tokenizer = Tokenizer.from_file("./TinyStories_tokenizer.json")


# In[21]:


from tokenizers import decoders
tokenizer.decoder = decoders.WordPiece()


# In[22]:


from components.generatetext import generate_text


# In[ ]:





# In[23]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
litmodel.model.to(device)


# In[ ]:





# In[24]:


get_ipython().run_cell_magic('time', '', 'litmodel.eval()\nstarting_text = "Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup. He wants to share it with Jane. “Jane, do you want some soup?” Tom asks. “Yes, please. It looks yummy,” Jane says. Tom pours some soup into two bowls. He gives one bowl to Jane. Jane takes a spoonful of soup, but then she makes a face. The soup is"\ntext = generate_text(litmodel.model, tokenizer, starting_text, 512, device, topk=3, temperature=1)\nprint("text: ", text)\n')


# In[25]:


get_ipython().run_cell_magic('time', '', 'litmodel.eval()\nstarting_text = "Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup. He wants to share it with Jane. “Jane, do you want some soup?” Tom asks. “Yes, please. It looks yummy,” Jane says. Tom pours some soup into two bowls. He gives one bowl to Jane. Jane takes a spoonful of soup, but then she makes a face. The soup is"\ntext = generate_text(litmodel.model, tokenizer, starting_text, 512, device, topk=3, temperature=1)\nprint("text: ", text)\n')


# In[26]:


get_ipython().run_cell_magic('time', '', 'litmodel.eval()\nstarting_text = "Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup. He wants to share it with Jane. “Jane, do you want some soup?” Tom asks. “Yes, please. It looks yummy,” Jane says. Tom pours some soup into two bowls. He gives one bowl to Jane. Jane takes a spoonful of soup, but then she makes a face. The soup is"\ntext = generate_text(litmodel.model, tokenizer, starting_text, 512, device, topk=3, temperature=1)\nprint("text: ", text)\n')


# In[27]:


get_ipython().run_cell_magic('time', '', 'litmodel.eval()\nstarting_text = "Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup. He wants to share it with Jane. “Jane, do you want some soup?” Tom asks. “Yes, please. It looks yummy,” Jane says. Tom pours some soup into two bowls. He gives one bowl to Jane. Jane takes a spoonful of soup, but then she makes a face. The soup is"\ntext = generate_text(litmodel.model, tokenizer, starting_text, 512, device, topk=3, temperature=1)\nprint("text: ", text)\n')


# In[ ]:




