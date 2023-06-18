# https://github.com/dh1105/Sentence-Entailment/blob/main/Sentence_Entailment_BERT.ipynb

import typing as t
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers.utils.logging as transformers_logging
import typer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification  # type: ignore
from transformers.modeling_outputs import SequenceClassifierOutput

from argument_nli.config import config
from argument_nli.model import AnnotationDataset
from argument_nli.train.model import EntailmentDataset

transformers_logging.set_verbosity_error()


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
) -> t.Tuple[np.float_, np.float_]:
    model.train()
    batch: t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]
    losses = []
    accuracies = []

    context = torch.no_grad if test else nullcontext

    with context():
        with typer.progressbar(
            data_loader,
            item_show_func=lambda _: epoch_progress(losses, accuracies),
            label="Evaluating" if test else "Training",
        ) as batches:
            # for batch in (pbar := tqdm(data_loader)):
            # pbar.set_description(epoch_progress(losses, accuracies))
            for batch in batches:
                optimizer.zero_grad()
                model_params = {
                    key: value.to(config.model.device)
                    for key, value in batch[0].items()
                }
                labels = batch[1].to(config.model.device)
                output: SequenceClassifierOutput = model(**model_params, labels=labels)

                if output.loss is not None:
                    # For DataParallel, the loss has to be aggregated
                    loss = output.loss.mean()

                    if not test:
                        loss.backward()
                        optimizer.step()

                    losses.append(loss.item())
                    accuracies.append(multi_acc(output.logits, labels).item())

    acc = np.mean(accuracies)
    loss = np.mean(losses)

    typer.echo(f"Mean accuracy: {acc}, Mean loss: {loss}")

    return acc, loss


dataset = AnnotationDataset()

for file in Path(config.model.train.dataset_path).glob(
    config.model.train.dataset_pattern
):
    dataset.update(AnnotationDataset.open(file))

train_data = EntailmentDataset(dataset.train).get_data_loader()
test_data = EntailmentDataset(dataset.test).get_data_loader()

model: torch.nn.Module = AutoModelForSequenceClassification.from_pretrained(
    config.model.pretrained, num_labels=3
)
model = torch.nn.DataParallel(model)  # type: ignore
model = model.to(config.model.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.learning_rate)

for epoch in range(config.model.epochs):
    typer.echo(f"Epoch {epoch+1}")

    train_acc, train_loss = train_model(model, train_data, optimizer, test=False)
    test_acc, test_loss = train_model(model, train_data, optimizer, test=True)

torch.save(model.state_dict(), config.model.path)
