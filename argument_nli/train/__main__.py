# https://github.com/dh1105/Sentence-Entailment/blob/main/Sentence_Entailment_BERT.ipynb

import gzip
import json
import typing as t
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import typer
from argument_nli.config import config
from argument_nli.convert import Annotation, AnnotationDataset
from argument_nli.train.model import EntailmentDataset
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


def multi_acc(y_pred: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
    return (
        torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test
    ).sum().float() / float(y_test.size(0))


def epoch_progress(losses: t.Sequence[float], accuracies: t.Sequence[float]) -> str:
    loss = losses[-1] if losses else "N/A"
    acc = accuracies[-1] if accuracies else "N/A"

    return f"Accuracy: {acc}, Loss: {loss}"


def train_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    test: bool,
) -> t.Tuple[float, float]:
    model.train()
    batch: t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]
    losses = []
    accuracies = []

    context = nullcontext

    if test:
        context = torch.no_grad

    with context():
        with typer.progressbar(
            data_loader,
            item_show_func=lambda _: epoch_progress(losses, accuracies),
            label="Evaluating" if test else "Training",
        ) as batches:
            for batch in batches:
                optimizer.zero_grad()
                model_params = {
                    key: value.to(config.model.device)
                    for key, value in batch[0].items()
                }
                labels = batch[1].to(config.model.device)
                output: SequenceClassifierOutput = model(**model_params, labels=labels)

                if output.loss:
                    if not test:
                        output.loss.backward()
                        optimizer.step()

                    losses.append(output.loss.item())
                    accuracies.append(multi_acc(output.logits, labels).item())

    acc = np.mean(accuracies)
    loss = np.mean(losses)

    typer.echo(f"Mean accuracy: {acc}, Mean loss: {loss}")

    return acc, loss


dataset = AnnotationDataset()

for file in config.path.datasets:
    with gzip.open(file, "rt", encoding="utf-8") as f:
        dataset.extend(AnnotationDataset.from_dict(json.load(f)))

train_data = EntailmentDataset(dataset.train).get_data_loader()
test_data = EntailmentDataset(dataset.test).get_data_loader()

model: torch.nn.Module = BertForSequenceClassification.from_pretrained(
    config.model.pretrained, num_labels=3
)
model.to(config.model.device)

optimizer = t.cast(
    torch.optim.AdamW,
    AdamW(model.parameters(), lr=config.model.learning_rate, correct_bias=False),
)

for epoch in range(config.model.epochs):
    typer.echo(f"Epoch {epoch+1}")

    train_acc, train_loss = train_model(model, train_data, optimizer, test=False)
    test_acc, test_loss = train_model(model, train_data, optimizer, test=True)

torch.save(model.state_dict(), config.path.model)
