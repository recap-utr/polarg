import torch
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup

from polarg.config import config
from polarg.model.annotation import label2proto
from polarg.model.dataset import BatchType, BatchTypeX


class EntailmentModule(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.pretrained, num_labels=3
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=3, top_k=1)

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
        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)

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

        return {enum: probabilities[label.value] for label, enum in label2proto.items()}

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
