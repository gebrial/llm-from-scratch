#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[189]:


import torch

def generate_next_token(tokens, model, device):
    logits = model(tokens)
    tokens_output = torch.argmax(logits, dim=-1)
    return tokens_output[0][-1]

def generate_tokens(model, start_tokens, max_length, device, eot_token):
    final_tokens = torch.full(size=(1, max_length), fill_value=eot_token, device=device)

    idx = len(start_tokens)
    final_tokens[0][:idx] = start_tokens
    while idx < max_length:
        tokens = final_tokens[:, :idx]
        next_token = generate_next_token(tokens, model, device)
        final_tokens[0][idx] = next_token
        idx += 1

        if next_token == eot_token:
            break

    return final_tokens


def generate_text(model, tokenizer, start_text, max_length, device, eot_string="<|endoftext|>"):
    """
    Ensure tokenizer decode is set to tokenizers.decoders.WordPiece() for best results
    max_length refers to number of tokens
    device can be either "cpu" or "cuda"
    """
    eot_token = tokenizer.encode(eot_string).ids[0]
    start_tokens = tokenizer.encode(start_text).ids
    start_tokens = torch.tensor(start_tokens).to(device)
    tokens = generate_tokens(model, start_tokens, max_length, device, eot_token)
    return tokenizer.decode(tokens[0].tolist())

