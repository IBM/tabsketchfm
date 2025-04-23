from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, r2_score
from .tabsketchfm import TabSketchFM
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchmetrics.classification import Accuracy
from .transformer_bert import TabularBertModel
from transformers import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from .SimpleModel import SimpleModel


class SequenceClassificationForTabularBertModel(nn.Module):
    def __init__(self, config, checkpoint, freeze):
        super(SequenceClassificationForTabularBertModel, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        if config.task_specific_params:
            self.model = TabularBertModel.from_pretrained(checkpoint)
        else:
            self.model = SimpleModel(config)
        if freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            token_position_ids: Optional[torch.Tensor] = None,
            value_ids: Optional[torch.FloatTensor] = None,
            minhash_vals: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and labels.dtype == torch.double:
            labels = labels.float()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            token_position_ids=token_position_ids,
            value_ids=value_ids,
            minhash_vals=minhash_vals,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
                    print('this is multi label classification')
                    
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs,
            attentions=outputs.attentions,
        )

class FinetuneTabSketchFM(TabSketchFM):
    def __init__(self, model_name_or_path, config, learning_rate= 2e-05, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, freeze=False, model_type='classification', task='fine-tune-table-similarity', num_labels=2):
        super().__init__(model_name_or_path, learning_rate, adam_beta1, adam_beta2, adam_epsilon)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.config = config
        self.model_type = model_type
        self.config.num_labels = num_labels
        self.num_labels = num_labels
        if model_type == 'classification' and task == 'fine-tune-table-similarity':
            self.model = SequenceClassificationForTabularBertModel(self.config, model_name_or_path, freeze)
        elif model_type == 'regression' and task == 'fine-tune-table-similarity':
            self.config.num_labels = 1
            self.model = SequenceClassificationForTabularBertModel(self.config, model_name_or_path, freeze)
        self.val_step_outputs = []
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        out = self.model(**data, labels=labels)
        loss = out.loss
        # logs metrics for each training_step,
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_step_outputs.extend(out.logits.cpu().numpy())
        self.val_step_targets.extend(labels.cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        if self.model_type == 'classification':
            self.compute_accuracy(torch.tensor(self.val_step_targets), torch.tensor(self.val_step_outputs), 'total_val_accuracy')  
        if self.model_type == 'regression':
            self.compute_regression(torch.tensor(self.val_step_targets),  torch.tensor(self.val_step_outputs), 'total_val_r2')

    def compute_regression(self, labels, logits, log_label):
        target = labels.view(-1).cpu()
        pred = logits.view(-1).cpu()
        r2 = r2_score(target, pred)
        r = pearsonr(target, pred)
        self.log(log_label, r2, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('r', r.statistic, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('p', r.pvalue, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def compute_accuracy(self, labels, logits, log_label):
        target, predicted_labels = self.get_predictions_labels(labels, logits, self.model_type)
        if self.num_labels == 2:
            accuracy = Accuracy(task='binary', average='weighted', num_labels=self.num_labels).to(self.device)
            acc = accuracy(predicted_labels, target)
        elif self.num_labels > 2:
            accuracy = Accuracy(task='multilabel', average='weighted', num_labels=self.num_labels).to(self.device)
            acc = accuracy(logits.to(self.device), labels.to(self.device))
        self.log(log_label, acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        f1_val = f1_score(target.cpu(), predicted_labels.cpu(), average='weighted', zero_division=1)
        self.log('f1', f1_val, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def get_predictions_labels(self, labels, logits, task_type):
        if task_type == 'classification':
            if self.num_labels == 2:
                print(logits)
                print(logits.shape)
                predicted_labels = logits.argmax(dim=1)
                target = labels.view(-1)
            elif self.num_labels > 2:
                predicted_labels = (torch.sigmoid(logits) > 0.5).float()
                predicted_labels = predicted_labels.view(-1, predicted_labels.size(-1))
                target = labels.view(-1, labels.size(-1))
        else:
            target = labels.view(-1)
            predicted_labels = logits.view(-1)

        return target, predicted_labels

    def test_step(self, batch, batch_idx):
        data, labels = batch
        out = self.model(**data, labels=labels)
        loss = out.loss
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.test_step_outputs.extend(out.logits.cpu().numpy())
        self.test_step_targets.extend(labels.cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        if self.model_type == 'classification':
            self.compute_accuracy(torch.tensor(self.test_step_targets), torch.tensor(self.test_step_outputs), 'total_test_accuracy')  
        if self.model_type == 'regression':
            self.compute_regression(torch.tensor(self.test_step_targets),  torch.tensor(self.test_step_outputs), 'total_test_r2')


