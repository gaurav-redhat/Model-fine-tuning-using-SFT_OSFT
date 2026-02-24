"""
Main fine-tuning script for LoRA / Adaptive-LoRA.

Usage:
    python finetune.py                         # defaults from config.py
    python finetune.py --adaptive --rank 16    # override via CLI
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LoRAConfig
from dataset import load_from_huggingface, build_dataloaders
from loss import LabelSmoothingLoss, LoRAFinetuneLoss
from model import LoraModel, LoraConfig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    cfg = LoRAConfig()

    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--num_labels", type=int, default=cfg.num_labels)
    parser.add_argument("--target_modules", nargs="+", default=cfg.target_modules)
    parser.add_argument("--rank", type=int, default=cfg.rank)
    parser.add_argument("--alpha", type=float, default=cfg.alpha)
    parser.add_argument("--lora_dropout", type=float, default=cfg.lora_dropout)
    parser.add_argument("--adaptive", action="store_true", default=cfg.adaptive)
    parser.add_argument("--no_adaptive", action="store_true")
    parser.add_argument("--gate_init", type=float, default=cfg.gate_init)
    parser.add_argument("--dataset_name", type=str, default=cfg.dataset_name)
    parser.add_argument("--text_column", type=str, default=cfg.text_column)
    parser.add_argument("--label_column", type=str, default=cfg.label_column)
    parser.add_argument("--max_length", type=int, default=cfg.max_length)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=cfg.max_grad_norm)
    parser.add_argument("--label_smoothing", type=float, default=cfg.label_smoothing)
    parser.add_argument("--gate_lambda", type=float, default=cfg.gate_lambda)
    parser.add_argument("--prune_threshold", type=float, default=cfg.prune_threshold)
    parser.add_argument("--prune_at_epoch", type=int, default=cfg.prune_at_epoch)
    parser.add_argument("--output_dir", type=str, default=cfg.output_dir)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--fp16", action="store_true", default=cfg.fp16)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--log_every", type=int, default=cfg.log_every)

    args = parser.parse_args()

    if args.no_adaptive:
        args.adaptive = False

    # Build config from parsed args, dropping the helper flag
    args_dict = vars(args)
    filtered = {}
    for key, value in args_dict.items():
        if key != "no_adaptive":
            filtered[key] = value

    return LoRAConfig(**filtered)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        # Move batch to device
        moved_batch = {}
        for key, value in batch.items():
            moved_batch[key] = value.to(device)

        labels = moved_batch.pop("labels")
        outputs = model(**moved_batch)
        logits = outputs.logits

        batch_loss = loss_fn(logits, labels)
        total_loss += batch_loss.item() * labels.size(0)

        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    val_loss = total_loss / total
    val_acc = correct / total

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(cfg):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config to disk
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    # ---- Tokenizer & Data ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = load_from_huggingface(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        max_length=cfg.max_length,
        split="train",
    )
    val_ds = load_from_huggingface(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        max_length=cfg.max_length,
        split="test",
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # ---- Model ----
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
    )

    lora_cfg = LoraConfig(
        rank=cfg.rank,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_rslora=True,
        adaptive=cfg.adaptive,
        gate_init=cfg.gate_init,
    )

    model = LoraModel(base_model, lora_cfg)
    model.to(device)

    # ---- Loss ----
    if cfg.label_smoothing > 0:
        task_loss_fn = LabelSmoothingLoss(cfg.num_labels, cfg.label_smoothing)
    else:
        task_loss_fn = nn.CrossEntropyLoss()

    if cfg.adaptive:
        gate_lambda = cfg.gate_lambda
    else:
        gate_lambda = 0.0

    criterion = LoRAFinetuneLoss(
        task_loss_fn=task_loss_fn,
        gate_lambda=gate_lambda,
    )

    # ---- Optimizer & Scheduler ----
    trainable_params = []
    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=cfg.warmup_ratio,
    )

    scaler = GradScaler(enabled=cfg.fp16)

    # ---- Train ----
    print(f"\nStarting training for {cfg.epochs} epochs on {device}")
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):

            # Move batch to device
            moved_batch = {}
            for key, value in batch.items():
                moved_batch[key] = value.to(device)

            labels = moved_batch.pop("labels")

            with autocast(enabled=cfg.fp16):
                outputs = model(**moved_batch)
                logits = outputs.logits

                if cfg.adaptive:
                    gate_reg = model.gate_regularization_loss()
                else:
                    gate_reg = None

                losses = criterion(logits, labels, gate_reg)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += losses["task_loss"].item()
            global_step += 1

            if global_step % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                task_loss_val = losses["task_loss"].item()
                msg = f"  [step {global_step}] task_loss={task_loss_val:.4f}"

                if "gate_reg" in losses:
                    gate_reg_val = losses["gate_reg"].item()
                    msg += f"  gate_reg={gate_reg_val:.6f}"

                msg += f"  lr={lr:.2e}"
                print(msg)

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{cfg.epochs}  "
            f"avg_loss={avg_loss:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # ---- Adaptive pruning ----
        if cfg.adaptive and cfg.prune_at_epoch == epoch:
            print(f"\n  Pruning adaptive ranks (threshold={cfg.prune_threshold})")
            results = model.prune_adaptive(cfg.prune_threshold)
            for name, pruned in results.items():
                print(f"    {name}: pruned {pruned} ranks")

        # ---- Validation ----
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_acc"]
            print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = str(out_dir / "best_lora.safetensors")
                model.save_model(save_path, merge_weights=False)
                print(f"  -> Saved best checkpoint (acc={best_val_acc:.4f})")

    # ---- Final save ----
    final_path = str(out_dir / "last_lora.safetensors")
    model.save_model(final_path, merge_weights=False)
    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoints saved in {out_dir}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
