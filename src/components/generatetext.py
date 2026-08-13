import torch


def _generate_next_token(tokens, model, device, temperature=1, topk=1):
    if temperature == 0:
        temperature = 1
        topk = 1
    if topk < 1:
        raise ValueError("topk must be >= 1")

    num_tokens = tokens.size(1)
    full_attn_mask = torch.ones((1, num_tokens, num_tokens), device=device)
    positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand_as(tokens)

    logits = model([tokens, full_attn_mask, positions])[0][-1] / temperature
    top_logits, top_pos = torch.topk(logits, topk)
    top_probs = torch.softmax(top_logits, dim=-1)
    next_token_pos = torch.multinomial(top_probs, num_samples=1)
    return top_pos[next_token_pos]


def generate_text(model, tokenizer, start_text, max_length, device, eot_string="<|endoftext|>",
                   temperature=1, topk=1, output_only=False):
    """
    Greedy/top-k autoregressive generation.

    max_length refers to number of tokens. topk must be >= 1. Set output_only=True
    to get back only the completion, without the prompt.
    """
    eot_token = tokenizer.encode(eot_string).ids[0]
    start_tokens = torch.tensor(tokenizer.encode(start_text).ids, device=device)
    start_tokens_len = start_tokens.size(0)

    final_tokens = torch.full((1, max_length), fill_value=eot_token, device=device)
    final_tokens[0, :start_tokens_len] = start_tokens

    idx = start_tokens_len
    hit_eot = False
    while idx < max_length:
        next_token = _generate_next_token(final_tokens[:, :idx], model, device, temperature, topk)
        final_tokens[0, idx] = next_token
        idx += 1
        if next_token == eot_token:
            hit_eot = True
            break

    # drop the trailing end-of-text token, if generation stopped because of one
    end = idx - 1 if hit_eot else idx
    tokens = final_tokens[0, start_tokens_len:end] if output_only else final_tokens[0, :end]
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=False)


def beam_search(model, tokenizer, start_text, max_beams=5, max_tokens=512, eot_token="<|endoftext|>"):
    """
    Returns up to max_beams completions that reached <|endoftext|>.

    In practice, beams tend to differ only in their last few tokens -- top_p_sampling
    below produces more varied, higher-quality completions for this model.
    """
    device = next(model.parameters()).device
    eot_token_id = tokenizer.encode(eot_token).ids[0]
    start_tokens = torch.tensor(tokenizer.encode(start_text).ids, device=device)

    beams = start_tokens.unsqueeze(0).repeat(max_beams, 1)
    beam_scores = torch.zeros(max_beams, device=device)
    beam_scores[0] = 1.0  # start with one active beam
    completed_beams = []

    for _ in range(max_tokens - start_tokens.size(0)):
        num_tokens = beams.size(1)
        full_attn_mask = torch.ones((max_beams, num_tokens, num_tokens), device=device)
        positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand_as(beams)

        with torch.no_grad():
            logits = model([beams, full_attn_mask, positions])[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        adjusted_scores = beam_scores.unsqueeze(1) + log_probs
        top_scores, top_indices = adjusted_scores.view(-1).topk(max_beams)
        beam_indices = top_indices // log_probs.size(-1)
        token_indices = top_indices % log_probs.size(-1)

        beams = torch.cat([beams[beam_indices], token_indices.unsqueeze(1)], dim=1)
        beam_scores = top_scores

        completed = token_indices == eot_token_id
        if completed.any():
            for i in range(max_beams):
                if completed[i]:
                    completed_beams.append(beams[i].cpu())
                    beam_scores[i] = float("-inf")  # disable completed beams

        if len(completed_beams) >= max_beams:
            break

    return [tokenizer.decode(beam.tolist()[:-1], skip_special_tokens=False) for beam in completed_beams]


def top_p_sampling(model, tokenizer, prompt, top_p=0.95, temperature=1.0, max_length=512,
                    eot_token="<|endoftext|>"):
    """
    Nucleus sampling with temperature. Empirically produces more varied, higher-quality
    completions than beam_search for this model.
    """
    device = next(model.parameters()).device
    eot_token_id = tokenizer.encode(eot_token).ids[0]

    prompt_tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(prompt_tokens, device=device).unsqueeze(0)

    for _ in range(max_length - len(prompt_tokens)):
        num_tokens = input_ids.size(1)
        full_attn_mask = torch.ones((1, num_tokens, num_tokens), device=device)
        positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand_as(input_ids)

        with torch.no_grad():
            logits = model([input_ids, full_attn_mask, positions])[:, -1, :]

        logits = (logits / temperature).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative_probs <= top_p
        keep_mask[..., 0] = True
        filtered_probs = sorted_probs * keep_mask.float()
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

        sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
        sampled_token = sorted_indices.gather(-1, sampled_idx).unsqueeze(0)
        input_ids = torch.cat([input_ids, sampled_token], dim=-1)
        if sampled_token.item() == eot_token_id:
            break

    output_tokens = input_ids.squeeze(0).tolist()
    # drop the trailing end-of-text token, if generation stopped because of one
    if output_tokens[-1] == eot_token_id:
        output_tokens = output_tokens[:-1]
    return tokenizer.decode(output_tokens, skip_special_tokens=False)
