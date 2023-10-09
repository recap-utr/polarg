import transformers.utils.logging as transformers_logging
from lightning import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from argent.config import config
from argent.model import EntailmentDataModule, EntailmentModule

transformers_logging.set_verbosity_error()


datamodule = EntailmentDataModule()

logger = WandbLogger(project="argent")
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
module = EntailmentModule()

trainer.fit(module, datamodule=datamodule)
trainer.test(datamodule=datamodule, ckpt_path="best")  # or last

trainer.save_checkpoint(config.model.path)
