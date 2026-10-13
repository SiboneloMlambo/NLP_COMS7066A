from __future__ import annotations

import copy
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SequenceDataset, make_autoregressive_inputs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_logits(model, context, targets):
    inputs = make_autoregressive_inputs(context, targets)
    logits = model(inputs)
    start = context.size(1) - 1
    return logits[:, start:start + targets.size(1), :]


@torch.no_grad()
def evaluate(model, dataset, batch_size: int, device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = total_tokens = exact = first_correct = 0
    all_pred, all_true = [], []
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for context, targets in loader:
        context, targets = context.to(device), targets.to(device)
        logits = target_logits(model, context, targets)
        total_loss += loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        pred = logits.argmax(-1)
        total_tokens += targets.numel()
        exact += (pred == targets).all(dim=1).sum().item()
        first_correct += (pred[:, 0] == targets[:, 0]).sum().item()
        all_pred.append(pred.cpu())
        all_true.append(targets.cpu())
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "token_accuracy": torch.cat(all_pred).eq(torch.cat(all_true)).float().mean().item(),
        "first_value_accuracy": first_correct / len(dataset),
        "exact_accuracy": exact / len(dataset),
        "predictions": torch.cat(all_pred).numpy(),
        "targets": torch.cat(all_true).numpy(),
    }


def fit(model, train_data, val_data, cfg: dict, device: torch.device) -> list[dict]:
    model.to(device)
    loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    history, best_loss, best_state, stale = [], float("inf"), None, 0
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, count, correct = 0.0, 0, 0
        for context, targets in loader:
            context, targets = context.to(device), targets.to(device)
            logits = target_logits(model, context, targets)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            running += loss.item() * targets.numel()
            count += targets.numel()
            correct += (logits.argmax(-1) == targets).sum().item()
        val = evaluate(model, val_data, cfg["batch_size"], device)
        row = {"epoch": epoch, "step": step, "train_loss": running / count,
               "train_token_accuracy": correct / count, "val_loss": val["loss"],
               "val_first_value_accuracy": val["first_value_accuracy"],
               "val_token_accuracy": val["token_accuracy"],
               "val_accuracy": val["exact_accuracy"]}
        history.append(row)
        if cfg.get("verbose", False):
            print(f"  epoch {epoch:3d} step {step:6d} train_loss {row['train_loss']:.3f} "
                  f"val_loss {row['val_loss']:.3f} val_first_value_acc "
                  f"{row['val_first_value_accuracy']:.3f}", flush=True)
        if val["loss"] < best_loss - 1e-5:
            best_loss, stale = val["loss"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def generate_targets(model, contexts: np.ndarray, target_length: int, batch_size: int,
                     device: torch.device, temperature: float = 1.0) -> np.ndarray:
    """Sample synthetic labels from a previous-generation model."""
    model.eval()
    outputs = []
    for start in range(0, len(contexts), batch_size):
        current = torch.as_tensor(contexts[start:start + batch_size], device=device)
        sampled = []
        for _ in range(target_length):
            logits = model(current)[:, -1, :]
            if temperature <= 0:  # greedy decoding
                token = logits.argmax(-1, keepdim=True)
            else:
                token = torch.multinomial((logits / temperature).softmax(-1), 1)
            sampled.append(token)
            current = torch.cat([current, token], dim=1)
        outputs.append(torch.cat(sampled, dim=1).cpu().numpy())
    return np.concatenate(outputs)

