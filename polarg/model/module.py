# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/cross-encoder/training_nli.py
# https://github.com/vibhor98/GraphNLI/blob/main/Baselines/sentence-bert/sbert_training.py

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup

from polarg.config import config
from polarg.model.annotation import label_to_proto
from polarg.model.dataset import BatchType, BatchTypeX


class EntailmentModule(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        # The CrossEncoder from SBERT does the same thing but is not compatible with lightning
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.pretrained, num_labels=len(label_to_proto)
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = (
            BinaryAccuracy()
            if len(label_to_proto) == 2
            else MulticlassAccuracy(num_classes=len(label_to_proto), top_k=1)
        )

    def forward(self, x: BatchTypeX):
        return self.model(**x)

    def _step(self, batch: BatchType, batch_idx: int, stage: str):
        x, y = batch

        # The predict phase does not have a label, this is just checked to be sure here.
        assert y is not None

        output: SequenceClassifierOutput = self(x)
        logits = output.logits
        predictions = logits.argmax(dim=1)

        loss = self.loss(logits, y)
        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)

        if stage == "training":
            return loss

        accuracy = self.accuracy(predictions, y)
        self.log(f"{stage}_accuracy", accuracy, on_epoch=True, sync_dist=True)

        return {"loss": loss, "accuracy": accuracy}

    def training_step(self, *args, **kwargs):
        return self._step(*args, **kwargs, stage="training")

    def validation_step(self, *args, **kwargs):
        return self._step(*args, **kwargs, stage="validation")

    def test_step(self, *args, **kwargs):
        return self._step(*args, **kwargs, stage="test")

    def predict_step(self, batch: BatchType, batch_idx: int):
        x, _ = batch
        output: SequenceClassifierOutput = self(x)
        probabilities: list[float] = F.softmax(output.logits, dim=1).flatten().tolist()

        return {
            enum: probabilities[label.value] for label, enum in label_to_proto.items()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.model.train.learning_rate
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.model.train.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
