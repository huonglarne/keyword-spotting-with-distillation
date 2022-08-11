import pytorch_lightning as pl
import torch
import torch.nn
import torchmetrics
from torchmetrics import Accuracy

from torch.nn import functional as F
import torch
from torch import nn

from argparse import ArgumentParser

from src.constants import LEARNING_RATE, NEW_SAMPLE_RATE, ORIGINAL_SAMPLE_RATE
import torchaudio

class BaseTorchLightlingWrapper(pl.LightningModule):
    def __init__(self, core_model, learning_rate=LEARNING_RATE):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.core_model = core_model
        self.accuracy = Accuracy()
        self.transform = torchaudio.transforms.Resample(orig_freq=ORIGINAL_SAMPLE_RATE, new_freq=NEW_SAMPLE_RATE)

    # will be used during inference
    def forward(self, x):
        embedding = self.core_model(x)
        return embedding
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform(x)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform(x)
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform(x)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        return optimizer