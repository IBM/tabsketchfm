from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from .transformer_bert import TabularBertForMaskedLM
from transformers import AutoConfig
from transformers.optimization import AdamW


class TabSketchFM(pl.LightningModule):
    def __init__(self, model_name_or_path, learning_rate, adam_beta1, adam_beta2, adam_epsilon):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = TabularBertForMaskedLM(self.config)

        print("Model created! %d parameters in core network." % sum([p.numel() for p in self.model.parameters()]))

    def forward(self, x):
        return self.model(x).logits

    def log_output(self, labels, logits, loss):
        # tensorboard
        logs = {"train_loss": loss}
        # identifying number of correct predections in a given batch
        correct = logits.view(-1, self.config.vocab_size).argmax(dim=1).eq(labels.view(-1)).sum().item()
        # identifying total number of labels in a given batch
        total = len(labels)
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total
        }
        return batch_dictionary

    def log_output_detailed(self, labels, logits, loss, batch_idx):
        return {'loss': loss, 'pred_labels': logits.view(-1, self.config.vocab_size).argmax(dim=1),
                'labels': labels.view(-1), 'idx': batch_idx}

    def epochMetrics(self, epochOutputs):
        epochPreds = []
        trueLabels = []
        totLoss = 0
        for out in epochOutputs:
            epochPreds = np.append(epochPreds, out['pred_labels'].cpu())
            trueLabels = np.append(trueLabels, out['labels'].cpu())
            totLoss += out['loss'].cpu()

        totLoss /= trueLabels.size
        acc = np.mean(epochPreds == trueLabels)
        return totLoss, acc

    def training_step(self, batch, batch_idx):
        data, labels = batch
        out = self.model(**data, labels=labels)
        loss = out.loss 
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        out = self.model(**data, labels=labels)
        loss = out.loss
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        out = self.model(**data, labels=labels)
        loss = out.loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.trainer.model.parameters(),
                        lr=self.learning_rate,
                        betas=(self.hparams.adam_beta1,
                                self.hparams.adam_beta2),
                        eps=self.hparams.adam_epsilon,)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser

