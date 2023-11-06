import typing as t
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BatchEncoding

from polarg.config import config
from polarg.model.annotation import Annotation, AnnotationDataset, EntailmentLabel

BatchTypeX = t.Dict[str, torch.Tensor]
BatchTypeY = t.Union[torch.Tensor, int]
BatchType = t.Tuple[BatchTypeX, BatchTypeY]

dataloader_args = {
    "batch_size": config.model.batch_size,
    "num_workers": config.model.dataloader_workers,
}


class EntailmentDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = AnnotationDataset()

        for file in Path(config.model.train.dataset_path).glob(
            config.model.train.dataset_pattern
        ):
            current_data = AnnotationDataset.open(file)

            if not config.model.train.include_neutral:
                current_data.remove_neutral()

            self.dataset.update(current_data)

    def train_dataloader(self):
        return DataLoader(
            EntailmentDataset(self.dataset.train),
            **dataloader_args,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            EntailmentDataset(self.dataset.test),
            **dataloader_args,
        )

    def val_dataloader(self):
        return DataLoader(
            EntailmentDataset(self.dataset.validation),
            **dataloader_args,
        )


class EntailmentDataset(Dataset):
    def __init__(self, annotations: t.Sequence[Annotation]):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained)
        self.annotations = annotations

    def _tokenize(self, premise: str, claim: str) -> BatchEncoding:
        return self.tokenizer(
            premise,
            claim,
            max_length=config.model.max_sequence_length,
            padding="max_length",
            truncation=True,  # add [PAD]
            add_special_tokens=True,  # add [CLS] and [SEP]
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _label(self, k: int) -> BatchTypeY:
        raw_label = self.annotations[k].label

        # for the predict phase, the label is `None``
        if raw_label is None:
            return -1

        return torch.tensor(EntailmentLabel(raw_label).value)

    def __getitem__(self, k: int) -> BatchType:
        ann = self.annotations[k]
        premise = ann.adus[ann.premise_id]
        claim = ann.adus[ann.claim_id]
        encoding = self._tokenize(premise, claim)

        return (
            {key: value.flatten() for key, value in encoding.items()},
            self._label(k),
        )
