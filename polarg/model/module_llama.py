import torch
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
from peft import (
    LoraConfig,  # type: ignore
    TaskType,  # type: ignore
    get_peft_model,  # type: ignore
    prepare_model_for_kbit_training,  # type: ignore
)
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from polarg.config import config
from polarg.model.annotation import label_to_proto
from polarg.model.dataset import BatchType, BatchTypeX


class EntailmentModule(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        # https://discuss.huggingface.co/t/llama-2-sequence-classification-much-lower-accuracy-on-inference-from-checkpoint-compared-to-model/54910
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.pretrained,
            num_labels=len(label_to_proto),
            quantization_config=q_config,
        )
        model.config.use_cache = False
        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=[
                "v_proj",
                "down_proj",
                "up_proj",
                "q_proj",
                "gate_proj",
                "k_proj",
                "o_proj",
            ],
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=len(label_to_proto), top_k=1
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
