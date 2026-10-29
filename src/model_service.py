"""
Load-once wrapper around a trained checkpoint, for callers that serve many
generate() calls from a long-running process (e.g. an API server) instead of
running once per process like scripts/generate.py does.

Usage:
    import _pathsetup  # noqa: F401 -- adds src/ to sys.path, as in scripts/
    from model_service import ModelService

    service = ModelService("checkpoints/run1/epoch=1-step=7044.ckpt", "data/tokenizer.json")
    story = service.generate("One day a girl walked into")
"""
import sys
from pathlib import Path

import torch
import tokenizers.decoders
from tokenizers import Tokenizer

from components.generatetext import beam_search, generate_text, top_p_sampling, top_p_token_ids

# LitGPTModel (the checkpoint's actual class) lives in scripts/train.py, which
# isn't on sys.path by default -- add it the same way _pathsetup.py adds src/.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train import LitGPTModel  # noqa: E402


class ModelService:
    """Loads a checkpoint + tokenizer once; generate() can then be called repeatedly."""

    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.model = self._load_model(checkpoint_path)

    @staticmethod
    def _load_tokenizer(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.decoder = tokenizers.decoders.ByteLevel()
        return tokenizer

    def _load_model(self, checkpoint_path):
        litmodel = LitGPTModel.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=self.device)
        litmodel.eval()
        litmodel.model.to(self.device)
        model = litmodel.model
        # Training saves compile=True in the checkpoint's hyperparameters, so Lightning
        # re-wraps the model in torch.compile on every load -- unwrap it here since
        # inference serves one-off, variable-length prompts torch.compile isn't suited for.
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def generate(self, prompt, method="top_p", max_length=512, temperature=1.0, top_p=0.95, topk=3, beams=5):
        """
        method: "top_p" (default, empirically the best completions -- see
        components/generatetext.py), "beam", or "greedy".
        """
        # A trailing space encodes as a lone "Ġ" token, which the model rarely saw
        # before a word in training -- it derails the continuation (e.g. "a , there").
        prompt = prompt.rstrip()
        if method == "top_p":
            return top_p_sampling(self.model, self.tokenizer, prompt, top_p=top_p,
                                   temperature=temperature, max_length=max_length)
        if method == "beam":
            completions = beam_search(self.model, self.tokenizer, prompt, max_beams=beams, max_tokens=max_length)
            return completions[0] if completions else ""
        if method == "greedy":
            return generate_text(self.model, self.tokenizer, prompt, max_length, self.device,
                                  temperature=temperature, topk=topk, output_only=True)
        raise ValueError(f"Unknown method: {method!r} (expected 'top_p', 'beam', or 'greedy')")

    def decoded_prompt(self, prompt):
        """
        The prompt as generate() puts it at the start of a story: stripped, encoded
        and decoded, which adds a leading space and drops anything the vocabulary
        cannot encode.
        """
        return self.tokenizer.decode(self.tokenizer.encode(prompt.rstrip()).ids, skip_special_tokens=False)

    def stream(self, prompt, max_length=512, temperature=1.0, top_p=0.95):
        """
        Yields the continuation's text one token at a time, top-p only. Unlike
        generate(), the prompt is not included: decoded_prompt(prompt) + the pieces
        joined is what generate() would return for the same seed.
        """
        prompt = prompt.rstrip()
        prompt_tokens = self.tokenizer.encode(prompt).ids
        eot_token_id = self.tokenizer.encode("<|endoftext|>").ids[0]
        # Each token is decoded on its own. A byte-level tokenizer can split one
        # character across two tokens, but this vocabulary has no non-ASCII tokens
        # (DECISIONS.md), so every token decodes to whole characters.
        for token_id in top_p_token_ids(self.model, prompt_tokens, eot_token_id, top_p=top_p,
                                        temperature=temperature, max_length=max_length):
            yield self.tokenizer.decode([token_id], skip_special_tokens=False)
