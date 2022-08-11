import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningDataModule

from src.constants import NUM_WORKERS, PIN_MEMORY, DATA_PATH, BATCH_SIZE
from typing import Callable, Optional
from torch.utils.data import Dataset

class SpeechCommandDataModule(LightningDataModule):
    def __init__(self, dataset: Dataset, collate_fn: Optional[Callable], data_dir= None, batch_size=None):
        super().__init__()
        self.dataset_obj = dataset
        self.collate_fn = collate_fn
        self.data_dir = data_dir or DATA_PATH
        self.batch_size = batch_size or BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.pin_memory = PIN_MEMORY


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj("training", self.data_dir),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj("validation", self.data_dir),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj("testing", self.data_dir),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )