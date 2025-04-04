#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[30]:


import torch

def generate_next_token(tokens, model, device, temperature=1, topk=1):
    if temperature == 0:
        temperature = 1
        topk = 1

    if topk < 1:
        raise ValueError("topk must be >= 1")

    logits = model(tokens)[0][-1] / temperature
    top_logits, top_pos = torch.topk(logits, topk)
    next_token_pos = torch.multinomial(top_logits, num_samples=1)
    return top_pos[next_token_pos]

def generate_tokens(model, start_tokens, max_length, device, eot_token, temperature=1, topk=1):
    final_tokens = torch.full(size=(1, max_length), fill_value=eot_token, device=device)

    idx = len(start_tokens)
    final_tokens[0][:idx] = start_tokens
    while idx < max_length:
        tokens = final_tokens[:, :idx]
        next_token = generate_next_token(tokens, model, device, temperature=temperature, topk=topk)
        final_tokens[0][idx] = next_token
        idx += 1

        if next_token == eot_token:
            break

    return final_tokens


def generate_text(model, tokenizer, start_text, max_length, device, eot_string="<|endoftext|>", temperature=1, topk=1):
    """
    Ensure tokenizer decode is set to tokenizers.decoders.WordPiece() for best results
    max_length refers to number of tokens
    device can be either "cpu" or "cuda"
    topk must be >= 1
    """
    eot_token = tokenizer.encode(eot_string).ids[0]
    start_tokens = tokenizer.encode(start_text).ids
    start_tokens = torch.tensor(start_tokens).to(device)
    tokens = generate_tokens(model, start_tokens, max_length, device, eot_token, temperature=temperature, topk=topk)
    return tokenizer.decode(tokens[0].tolist())


# In[ ]:




