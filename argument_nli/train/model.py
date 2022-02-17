import typing as t

import torch
from argument_nli.config import config
from argument_nli.convert import Annotation, EntailmentLabel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer  # type: ignore


class EntailmentDataset(Dataset):
    def __init__(self, annotations: t.Sequence[Annotation]):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            config.model.pretrained
        )
        self.label_dict = {
            EntailmentLabel.NEUTRAL: 0,
            EntailmentLabel.ENTAILMENT: 1,
            EntailmentLabel.CONTRADICTION: 2,
        }
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def _label(self, k: int) -> torch.Tensor:
        return torch.tensor(self.label_dict[self.annotations[k].label])

    def __getitem__(self, k: int) -> t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]:
        ann = self.annotations[k]

        encoding = self.tokenizer.encode_plus(  # type: ignore
            ann.premise,
            ann.claim,
            padding="max_length",  # TODO: default off
            max_length=config.model.max_sequence_length,
            truncation=True,  # TODO: default off
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return (
            {key: value.flatten() for key, value in encoding.items()},
            self._label(k),
        )

        # return EncodedAnnotation(
        #     encoding["input_ids"].flatten(),
        #     encoding["token_type_ids"].flatten(),
        #     encoding["attention_mask"].flatten(),
        #     torch.tensor(self.annotations[k].label),
        # )

    def get_data_loader(self):
        return DataLoader(self, shuffle=True, batch_size=config.model.train.batch_size)
