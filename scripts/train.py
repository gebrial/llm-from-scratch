"""
Train the GPT model on the packed TinyStories dataset produced by prepare_dataset.py.

Usage:
    python scripts/train.py --dataset data/packed_dataset --tokenizer data/tokenizer.json
"""
import argparse

import _pathsetup  # noqa: F401 -- adds src/ to sys.path before importing components

import lightning as L
import torch
import torch.nn as nn
from datasets import load_from_disk
from lightning.pytorch.callbacks import ModelCheckpoint
from tokenizers import Tokenizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from components.gptmodel import GPTModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/packed_dataset")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/run")
    parser.add_argument("--resume-from", default=None,
                         help="Checkpoint path to resume training from")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-batch-size", type=int, default=256,
                         help="Effective batch size via gradient accumulation")
    parser.add_argument("--max-lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")

    # model shape -- defaults match the "120M shallow" config the winning run used
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=24)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--weight-tying", action="store_true")

    return parser.parse_args()


class LitGPTModel(L.LightningModule):
    def __init__(self, trainer_config, gpt_config):
        super().__init__()
        self.save_hyperparameters()
        self.gpt_config = gpt_config
        self.trainer_config = trainer_config

        self.train_accuracy = []
        self.val_accuracy = []
        self.train_losses = []
        self.val_losses = []
        self.val_steps = []
        self.learning_rates = []
        self.batch_step = 0

    def _accuracy(self, output, expected):
        total_matching = (torch.argmax(output, dim=-1) == expected).sum().item()
        return total_matching / expected.numel()

    def _step(self, batch):
        x, y = batch["packed_inputs"][:, :-1], batch["packed_inputs"][:, 1:]
        attn_mask = batch["attention_mask"][:, :-1, :-1]
        positions = batch["padded_positions"][:, :-1]
        logits = self.model([x, attn_mask, positions])
        return logits, y

    def training_step(self, batch, batch_idx):
        self.batch_step += 1
        logits, y = self._step(batch)

        accuracy = self._accuracy(logits, y)
        self.log("accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.train_accuracy.append(accuracy)

        loss = self.loss(logits, y)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_losses.append(loss.item())

        self.learning_rates.append(self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_steps.append(self.batch_step)
        logits, y = self._step(batch)

        accuracy = self._accuracy(logits, y)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.val_accuracy.append(accuracy)

        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.val_losses.append(loss.item())
        return loss

    def loss(self, output, expected):
        return nn.functional.cross_entropy(output.flatten(0, 1), expected.flatten())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.trainer_config["max_lr"], weight_decay=0.1
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.trainer_config["max_lr"],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "monitor": "loss"},
        }

    def setup(self, stage):
        self.packed_dataset = load_from_disk(self.trainer_config["dataset"])
        self.packed_dataset.set_format("torch")

    def configure_model(self):
        if hasattr(self, "model"):
            return
        self.model = GPTModel(self.gpt_config)
        if self.trainer_config["compile"]:
            self.model = torch.compile(self.model, fullgraph=True)

    def _dataloader(self, split, shuffle):
        return DataLoader(
            self.packed_dataset[split],
            batch_size=self.trainer_config["batch_size"],
            shuffle=shuffle,
            num_workers=self.trainer_config["num_workers"],
            pin_memory=True,
            persistent_workers=self.trainer_config["num_workers"] > 0,
            prefetch_factor=2 if self.trainer_config["num_workers"] > 0 else None,
            drop_last=True,
        )

    def train_dataloader(self):
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._dataloader("validation", shuffle=False)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    vocab_size = Tokenizer.from_file(args.tokenizer).get_vocab_size()

    gpt_config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
        "weight_tying": args.weight_tying,
        "no_pos_emb": True,  # positions come from RoPE instead
    }

    trainer_config = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "max_lr": args.max_lr,
        "compile": args.compile,
        "grad_batches": max(1, args.target_batch_size // args.batch_size),
    }

    if args.resume_from:
        litmodel = LitGPTModel.load_from_checkpoint(checkpoint_path=args.resume_from)
    else:
        litmodel = LitGPTModel(trainer_config, gpt_config)

    trainer = L.Trainer(
        max_epochs=trainer_config["epochs"],
        logger=False,
        enable_progress_bar=True,
        accumulate_grad_batches=trainer_config["grad_batches"],
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(save_top_k=-1, every_n_epochs=1, dirpath=args.checkpoint_dir)
        ],
        precision=args.precision,
    )
    trainer.fit(model=litmodel)


if __name__ == "__main__":
    main()
