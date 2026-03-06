from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelBundle:
    model: nn.Module
    criterion: nn.Module
    batch: Any
    target: torch.Tensor
    optimizer_ctor: Any


def _resnet152_bundle(batch_size: int, device: str) -> ModelBundle:
    try:
        from torchvision.models import resnet152
    except Exception as e:
        raise RuntimeError(
            "ResNet-152 requested but torchvision is unavailable. Install torchvision first."
        ) from e

    num_classes = 1000
    model = resnet152(weights=None).to(device)
    criterion = nn.CrossEntropyLoss()

    batch = torch.randn(batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, num_classes, (batch_size,), device=device)

    def optimizer_ctor(params):
        return torch.optim.AdamW(params, lr=1e-4)

    return ModelBundle(
        model=model,
        criterion=criterion,
        batch=batch,
        target=target,
        optimizer_ctor=optimizer_ctor,
    )


def _bert_bundle(batch_size: int, device: str) -> ModelBundle:
    try:
        from transformers import BertConfig, BertForSequenceClassification
    except Exception as e:
        raise RuntimeError(
            "BERT requested but transformers is unavailable. Install transformers first."
        ) from e

    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=2,
    )
    model = BertForSequenceClassification(config).to(device)
    criterion = nn.CrossEntropyLoss()

    seq_len = 128
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
    labels = torch.randint(0, config.num_labels, (batch_size,), device=device)

    batch: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    class BertLossWrapper(nn.Module):
        def __init__(self, criterion: nn.Module):
            super().__init__()
            self.criterion = criterion

        def forward(self, out, target):
            logits = out.logits
            return self.criterion(logits, target)

    wrapped_criterion = BertLossWrapper(criterion)

    def optimizer_ctor(params):
        return torch.optim.AdamW(params, lr=2e-5)

    return ModelBundle(
        model=model,
        criterion=wrapped_criterion,
        batch=batch,
        target=labels,
        optimizer_ctor=optimizer_ctor,
    )


def build_model_bundle(model_name: str, batch_size: int, device: str) -> ModelBundle:
    name = model_name.lower().strip()
    if name == "resnet152":
        return _resnet152_bundle(batch_size=batch_size, device=device)
    if name == "bert":
        return _bert_bundle(batch_size=batch_size, device=device)
    raise ValueError(f"Unsupported model '{model_name}'. Use one of: resnet152, bert")
