import typing as t

import numpy as np
import torch
import torch.nn.functional as F
from arg_services.mining.v1beta.entailment_pb2 import EntailmentType
from transformers import (
    AutoModelForSequenceClassification,  # type: ignore
    AutoTokenizer,  # type: ignore
)
from transformers.modeling_outputs import SequenceClassifierOutput

from argument_nli.config import config


class EntailmentClassifier:
    def __init__(self):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            config.model.pretrained
        )

        self.model: torch.nn.Module = (
            AutoModelForSequenceClassification.from_pretrained(
                config.model.pretrained, num_labels=3
            )
        )
        self.model = torch.nn.DataParallel(self.model)  # type: ignore
        self.model.load_state_dict(
            torch.load(
                config.model.path,
                map_location=torch.device(config.model.device),
            )
        )
        self.model.eval()
        self.model.to(config.model.device)

        self.label_dict: t.Dict[int, EntailmentType.ValueType] = {
            0: EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
            1: EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
            2: EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
        }

    def predict(
        self, premise: str, claim: str
    ) -> t.Tuple[EntailmentType.ValueType, t.Dict[EntailmentType.ValueType, float]]:
        # we assume batch size to be 1, thus we can ignore padding -> improves performance
        encoding = self.tokenizer.encode_plus(  # type: ignore
            premise,
            claim,
            padding=False,  # TODO: default off
            max_length=config.model.max_sequence_length,
            truncation=True,  # TODO: default off
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_params = {
            key: value.to(config.model.device) for key, value in encoding.items()
        }

        with torch.no_grad():
            output: SequenceClassifierOutput = self.model(**model_params)
            probabilities: t.List[float] = (
                F.softmax(output.logits, dim=1).flatten().tolist()
            )
            prediction_idx = np.argmax(probabilities)
            prediction = self.label_dict[prediction_idx]
            probabilities_map = {
                enum: probabilities[idx] for idx, enum in self.label_dict.items()
            }

        return prediction, probabilities_map
