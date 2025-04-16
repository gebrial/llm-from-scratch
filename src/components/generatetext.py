#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import torch

def generate_next_token(tokens, model, device, temperature=1, topk=1):
    if temperature == 0:
        temperature = 1
        topk = 1

    if topk < 1:
        raise ValueError("topk must be >= 1")

    logits = model([tokens, None])[0][-1] / temperature
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


import torch

def generate_next_token_with_attn(tokens, model, device, temperature=1, topk=1):
    if temperature == 0:
        temperature = 1
        topk = 1

    if topk < 1:
        raise ValueError("topk must be >= 1")

    full_attn_mask = torch.ones((1, len(tokens[0]), len(tokens[0])), device=device)
    logits = model([tokens, full_attn_mask])
    logits = logits[0][-1] / temperature
    top_logits, top_pos = torch.topk(logits, topk)
    next_token_pos = torch.multinomial(top_logits, num_samples=1)
    return top_pos[next_token_pos]

def generate_tokens_with_attn(model, start_tokens, max_length, device, eot_token, temperature=1, topk=1):
    final_tokens = torch.full(size=(1, max_length), fill_value=eot_token, device=device)

    idx = len(start_tokens)
    final_tokens[0][:idx] = start_tokens
    while idx < max_length:
        tokens = final_tokens[:, :idx]
        next_token = generate_next_token_with_attn(tokens, model, device, temperature=temperature, topk=topk)
        final_tokens[0][idx] = next_token
        idx += 1

        if next_token == eot_token:
            break

    return final_tokens


def generate_text_with_attn(model, tokenizer, start_text, max_length, device, eot_string="<|endoftext|>", temperature=1, topk=1, output_only=False):
    """
    Ensure tokenizer decode is set to tokenizers.decoders.WordPiece() for best results
    max_length refers to number of tokens
    device can be either "cpu" or "cuda"
    topk must be >= 1
    """
    eot_token = tokenizer.encode(eot_string).ids[0]
    start_tokens = tokenizer.encode(start_text).ids
    start_tokens_len = len(start_tokens)
    start_tokens = torch.tensor(start_tokens).to(device)
    tokens = generate_tokens_with_attn(model, start_tokens, max_length, device, eot_token, temperature=temperature, topk=topk)
    if output_only:
        return tokenizer.decode(tokens[0][start_tokens_len:].tolist())
    return tokenizer.decode(tokens[0].tolist())


# In[ ]:


import torch

def generate_next_token_with_attn_positions(tokens, model, device, temperature=1, topk=1):
    if temperature == 0:
        temperature = 1
        topk = 1

    if topk < 1:
        raise ValueError("topk must be >= 1")

    full_attn_mask = torch.ones((1, len(tokens[0]), len(tokens[0])), device=device)
    positions = torch.arange(len(tokens[0]), device=device).unsqueeze(0).expand_as(tokens)

    logits = model([tokens, full_attn_mask, positions])
    logits = logits[0][-1] / temperature
    top_logits, top_pos = torch.topk(logits, topk)
    top_softmax = torch.softmax(top_logits, dim=-1)
    next_token_pos = torch.multinomial(top_softmax, num_samples=1)
    return top_pos[next_token_pos]

def generate_tokens_with_attn_positions(model, start_tokens, max_length, device, eot_token, temperature=1, topk=1):
    final_tokens = torch.full(size=(1, max_length), fill_value=eot_token, device=device)

    idx = len(start_tokens)
    final_tokens[0][:idx] = start_tokens
    while idx < max_length:
        tokens = final_tokens[:, :idx]
        next_token = generate_next_token_with_attn_positions(tokens, model, device, temperature=temperature, topk=topk)
        final_tokens[0][idx] = next_token
        idx += 1

        if next_token == eot_token:
            break

    return [final_tokens[0][:idx - 1]]


def generate_text_with_attn_positions(model, tokenizer, start_text, max_length, device, eot_string="<|endoftext|>", temperature=1, topk=1, output_only=False):
    """
    Ensure tokenizer decode is set to tokenizers.decoders.WordPiece() for best results
    max_length refers to number of tokens
    device can be either "cpu" or "cuda"
    topk must be >= 1
    """
    eot_token = tokenizer.encode(eot_string).ids[0]
    start_tokens = tokenizer.encode(start_text).ids
    start_tokens_len = len(start_tokens)
    start_tokens = torch.tensor(start_tokens).to(device)
    tokens = generate_tokens_with_attn_positions(model, start_tokens, max_length, device, eot_token, temperature=temperature, topk=topk)

    if output_only:
        return tokenizer.decode(tokens[0][start_tokens_len:].tolist(), skip_special_tokens=False)
    return tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)


# In[ ]:





# In[ ]:





# In[ ]:


def beam_search(
    model, 
    tokenizer, 
    start_text, 
    max_beams=5, 
    max_tokens=512,
    eot_token="<|endoftext|>"
):
    device = next(model.parameters()).device
    eot_token_id = tokenizer.encode(eot_token).ids[0]
    start_tokens = torch.tensor(tokenizer.encode(start_text).ids, device=device)

    # Initialize beams
    beams = start_tokens.unsqueeze(0).repeat(max_beams, 1)
    beam_scores = torch.zeros(max_beams, device=device)
    beam_scores[0] = 1.0  # Start with one active beam
    completed_beams = []

    for step in range(max_tokens - start_tokens.size(0)):
        num_tokens = beams.size(1)
        full_attn_mask = torch.ones((max_beams, num_tokens, num_tokens), device=device)
        positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand_as(beams)

        # Forward pass
        with torch.no_grad():
            logits = model([beams, full_attn_mask, positions])[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        # Get top candidates
        adjusted_scores = beam_scores.unsqueeze(1) + log_probs
        top_scores, top_indices = adjusted_scores.view(-1).topk(max_beams)
        beam_indices = top_indices // log_probs.size(-1)
        token_indices = top_indices % log_probs.size(-1)

        # Update beams and histories
        new_beams = torch.cat([beams[beam_indices], token_indices.unsqueeze(1)], dim=1)

        beams, beam_scores = new_beams, top_scores

        # Check for completion
        completed = token_indices == eot_token_id
        if completed.any():
            for i in range(max_beams):
                if completed[i]:
                    completed_beams.append(beams[i].cpu())
                    beam_scores[i] = float("-inf")  # Disable completed beams

        if len(completed_beams) >= max_beams:
            break

    return [tokenizer.decode(beam.tolist()[:-1], skip_special_tokens=False) for beam in completed_beams]


# In[ ]:





# In[ ]:


def top_p_sampling(
    model,
    tokenizer,
    prompt,
    top_p=0.95,
    temperature=1.0,
    max_length=512,
    eot_token="<|endoftext|>"
):
    device = next(model.parameters()).device
    eot_token_id = tokenizer.encode(eot_token).ids[0]

    prompt_tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(prompt_tokens, device=device).unsqueeze(0)

    for step in range(max_length - len(prompt_tokens)):
        num_tokens = input_ids.size(1)
        full_attn_mask = torch.ones((1, num_tokens, num_tokens), device=device)
        positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand_as(input_ids)

        # Forward pass
        with torch.no_grad():
            logits = model([input_ids, full_attn_mask, positions])[:, -1, :]

        logits = logits / temperature
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative_probs <= top_p
        keep_mask[..., 0] = 1
        filtered_probs = sorted_probs * keep_mask.float()
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
        sampled_token = sorted_indices.gather(-1, sampled_idx)
        sampled_token = sampled_token.unsqueeze(0)
        # print("input ids shape", input_ids.shape)
        # print("sampled token shape", sampled_token.shape)
        input_ids = torch.cat([input_ids, sampled_token], dim=-1)
        if sampled_token.item() == eot_token_id:
            break

    output_tokens = input_ids.squeeze(0).tolist()
    output_text = tokenizer.decode(output_tokens[:-1], skip_special_tokens=False)
    return output_text




# In[ ]:





# In[ ]:




