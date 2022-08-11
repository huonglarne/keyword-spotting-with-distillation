import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.data_utils.data_lightning_wrapper import SpeechCommandDataModule
from src.data_utils.datasets import AudioDataset
from src.model_utils.model_lightning_wrapper import BaseTorchLightlingWrapper
from src.model_utils.simple_conv import SimpleConv


if __name__ == "__main__":
    core_model = SimpleConv()

    pl.seed_everything(0)
    wandb_logger = WandbLogger(project="ViT_experiments")
    model = BaseTorchLightlingWrapper(core_model)

    data_module = SpeechCommandDataModule(AudioDataset, collate_fn=None)

    data_module.setup()

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, max_epochs=50, logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=50, logger=wandb_logger)

    # train, validate
    trainer.fit(model, data_module)