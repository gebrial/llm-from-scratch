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

# In[1]:


from components.gptmodel import GPTModel


# In[2]:


import lightning as L


# In[ ]:





# In[3]:


GPT_CONFIG_124M = {
    "vocab_size": 30002,
    "context_length": 256,
    "emb_dim": 512,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "weight_tying": True,
    "no_pos_emb": True
}


# In[4]:


import torch.nn as nn
import torch


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


class LitGPTModel(L.LightningModule):
    def __init__(self, GPTModel):
        super().__init__()
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

        # if self.batch_step == 5:
        #     self.x1 = x
        #     self.y1 = y
        #     self.logits1 = logits
            # print("x shape: ", x.shape, "y shape: ", y.shape, "logits shape: ", logits.shape)
            # print("x: ", x)
            # print("y: ", y)
            # print("logits: ", logits)

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
            self.parameters(), lr=1e-3, weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
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





# In[6]:


from components.data import create_dataloader_v3


# In[7]:


trainer_config = {
    "dataset_scale": 300,
    "batch_size": 32
}
trainer_config["grad_batches"] = 128 // trainer_config["batch_size"]


# In[8]:


def create_dataloader(text, train=True):
    return create_dataloader_v3(
        text,
        batch_size=trainer_config["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=train,
        shuffle=train,
        num_workers=11
    )


# In[9]:


get_ipython().run_cell_magic('time', '', '\ntrain_file = "../data/TinyStories/TinyStoriesV2-GPT4-train.txt"\nwith open(train_file, "r", encoding="utf-8") as f:\n    train_text = f.read()\n\ntrain_len = len(train_text)\ntrain_text = train_text[:train_len // trainer_config["dataset_scale"]]\ntrain_loader = create_dataloader(train_text)\n')


# In[10]:


val_file = "../data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
with open(val_file, "r", encoding="utf-8") as f:
    val_text = f.read()

val_len = len(val_text)
val_text = val_text[:val_len // trainer_config["dataset_scale"]]
val_loader = create_dataloader(val_text, train=False)


# In[ ]:





# In[11]:


model = GPTModel(GPT_CONFIG_124M)
litmodel = LitGPTModel(model)


# In[12]:


get_ipython().run_cell_magic('time', '', '\ntrainer = L.Trainer(max_epochs=1, enable_progress_bar=True, accumulate_grad_batches=trainer_config["grad_batches"])\ntrainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)\n')


# In[13]:


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(litmodel.train_losses, label="Training Loss", color="blue")
ax1.scatter(litmodel.val_steps, litmodel.val_losses, label="Validation Loss", color="red")
ax1.set_ylim(0, 20)
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


# In[14]:


plt.figure(figsize=(10, 5))
plt.plot(litmodel.train_accuracy, color="blue")
plt.scatter(litmodel.val_steps, litmodel.val_accuracy, color="red")


# In[ ]:




