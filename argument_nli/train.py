import transformers.utils.logging as transformers_logging
from lightning import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from argument_nli.config import config
from argument_nli.model import EntailmentDataModule, EntailmentModule

transformers_logging.set_verbosity_error()


datamodule = EntailmentDataModule()

logger = WandbLogger(project="argument-nli")
trainer = Trainer(
    logger=logger,
    max_epochs=config.model.train.max_epochs,
    accelerator="gpu",
    # strategy=DeepSpeedStrategy(
    #     stage=3,
    #     # offload_optimizer=True,
    #     # offload_parameters=True,
    # ),
)
model = EntailmentModule()

trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule, ckpt_path="best")  # or last

trainer.save_checkpoint(config.model.path)
