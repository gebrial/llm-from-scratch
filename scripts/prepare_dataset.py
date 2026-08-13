"""
Tokenize TinyStories and pack it into fixed-length training sequences.

Multiple short stories are greedily packed into each sequence (up to
context_length + 1 tokens, story order sorted by length for better packing
density) to avoid wasting compute on padding. An attention mask is stored
alongside each sequence so the model can't attend across story boundaries.

Usage:
    python scripts/prepare_dataset.py --tokenizer data/tokenizer.json --output data/packed_dataset
"""
import argparse

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from prepare_tokenizer import is_clean_story


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--output", default="data/packed_dataset")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--min-chars", type=int, default=418)
    parser.add_argument("--max-chars", type=int, default=2505)
    parser.add_argument("--num-proc", type=int, default=8)
    return parser.parse_args()


def pack_token_lists(stories, max_length):
    """Greedily bin-packs token lists into sequences no longer than max_length."""
    stories_len_sorted = sorted(stories["input_ids"], key=len, reverse=True)

    inputs = []
    token_positions = []

    for story in stories_len_sorted:
        story_length = len(story)

        if story_length >= max_length:
            story = story[:max_length]
            inputs.append(story)
            token_positions.append(list(range(len(story))))
            continue

        placed = False
        for input_seq, position_seq in zip(inputs, token_positions):
            if len(input_seq) + story_length <= max_length:
                input_seq.extend(story)
                position_seq.extend(range(story_length))
                placed = True
                break

        if not placed:
            inputs.append(story)
            token_positions.append(list(range(story_length)))

    return {"packed_inputs": inputs, "positions": token_positions}


def pad_sequences(example, max_length, padding_value):
    sequence = example["packed_inputs"]
    positions = example["positions"]
    padded_input = sequence + [padding_value] * (max_length - len(sequence))
    padded_positions = positions + [0] * (max_length - len(positions))
    return {
        "input_ids": padded_input[:max_length],
        "padded_positions": padded_positions[:max_length],
    }


def create_attention_mask(example, padding_value):
    """Blocks attention between tokens on either side of a story boundary."""
    input_ids = np.array(example["input_ids"])
    padding_indexes = np.where(input_ids == padding_value)[0]

    attention_mask = np.ones((len(input_ids), len(input_ids)), dtype=np.bool_)
    for padding_index in padding_indexes:
        attention_mask[:padding_index + 1, padding_index + 1:] = 0
        attention_mask[padding_index + 1:, :padding_index + 1] = 0

    return {"packed_inputs": example["input_ids"], "attention_mask": attention_mask}


def main():
    args = parse_args()
    max_length = args.context_length + 1  # +1 for the shifted prediction target

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot_token = tokenizer.encode("<|endoftext|>").ids[0]

    dataset = load_dataset("roneneldan/TinyStories")
    dataset = dataset.filter(lambda ex: is_clean_story(ex, args.min_chars, args.max_chars))

    def tokenize(examples):
        encodings = tokenizer.encode_batch_fast(examples["text"])
        return {"input_ids": [enc.ids + [eot_token] for enc in encodings]}

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=args.num_proc)

    dataset = dataset.map(
        lambda stories: pack_token_lists(stories, max_length),
        batched=True,
        remove_columns=["input_ids"],
    )

    dataset = dataset.map(
        lambda ex: pad_sequences(ex, max_length, eot_token),
        batched=False,
        remove_columns=["packed_inputs", "positions"],
        num_proc=args.num_proc,
    )

    dataset = dataset.map(
        lambda ex: create_attention_mask(ex, eot_token),
        batched=False,
        remove_columns=["input_ids"],
        num_proc=args.num_proc,
    )

    dataset.set_format("torch")
    dataset.save_to_disk(args.output)
    print(f"Saved packed dataset to {args.output}")
    print(dataset)


if __name__ == "__main__":
    main()
