import transformers.utils.logging as transformers_logging
from lightning import Trainer

from argument_nli.config import config
from argument_nli.model import EntailmentDataModule, EntailmentModule

transformers_logging.set_verbosity_error()


datamodule = EntailmentDataModule()

trainer = Trainer(max_epochs=config.model.train.max_epochs)
model = EntailmentModule()

trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule, ckpt_path="best")  # or last

trainer.save_checkpoint(config.model.path)
