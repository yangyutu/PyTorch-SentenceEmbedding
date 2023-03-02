import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import pytorch_lightning as pl
from models.pretrained_encoder import PretrainedSentenceEncoder
from torchmetrics import Accuracy
from models.utils import evaluate_stsb
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from models.utils import batch_to_device

# class FineTuneEncoder:
#     @classmethod
#     def load_enocder_weight(cls, ckpt):
#         ckpt = torch.load(ckpt_path)
#         model.load_state_dict(ckpt["state_dict"])


class ClassificationFinetuneEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        prediction_head: nn.Module,
        num_classes: int,
        config: Dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = encoder
        self.prediction_head = prediction_head
        self.config = config

        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def model_forward(self, batch):

        sentence_pairs = batch["sentence_pair"]
        sentence_list_1, sentence_list_2 = list(map(list, zip(*sentence_pairs)))
        embeddings_1 = self.model.encode(sentence_list_1, device=self.device)
        embeddings_2 = self.model.encode(sentence_list_2, device=self.device)

        logits = self.prediction_head([embeddings_1, embeddings_2])

        return logits

    def training_step(self, batch, batch_idx=0):

        logits = self.model_forward(batch)
        targets = batch["label"].to(self.device)
        loss = self.loss_fn(logits, targets.view(-1))

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.train_accuracy.update(logits, targets)
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log(
            "train_acc_epoch", self.train_accuracy.compute(), prog_bar=True, logger=True
        )
        self.train_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.get("lr", 1e-3))
        lr_warmup_steps = self.config.get("lr_warm_up_steps", 10000)

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["encoder_state_dict"] = self.model.state_dict()


class NPairContrastiveFinetuneEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        contrast_loss: nn.Module,
        config: Dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = encoder
        self.contrast_loss = contrast_loss
        self.config = config

    def model_forward(self, batch):
        # each batch can contain 2 or most lists of sentences,
        # first list is anchor, second list is positive, and (optional) third is selected negatives
        sentence_lists = batch
        embeddings = [
            self.model.encode(list(sentence_list), device=self.device)
            for sentence_list in sentence_lists
        ]

        loss = self.contrast_loss(embeddings)
        return loss

    def training_step(self, batch, batch_idx=0):

        loss = self.model_forward(batch)

        self.log(
            "train_loss",
            loss,
            batch_size=len(batch[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx=0):

        loss = self.model_forward(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_epoch_end(self, outs):

        stsb_result = evaluate_stsb(
            self.model, "/mnt/d/MLData/data/stsbenchmark.tsv", self.device
        )
        self.log(
            "Cosine-Similarity-Spearman_epoch",
            stsb_result["Cosine-Similarity-Spearman"],
            prog_bar=True,
            logger=True,
        )

        self.log(
            "Dot-Product-Similarity-Spearman_epoch",
            stsb_result["Dot-Product-Similarity-Spearman"],
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.get("lr", 1e-3))
        lr_warmup_steps = self.config.get("lr_warm_up_steps", 10000)

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["encoder_state_dict"] = self.model.state_dict()


class DenoisingAEFinetuneEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        pretrained_encoder_name: str,
        config: Dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = encoder
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_encoder_name)
        self._init_decoder(pretrained_encoder_name)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.config = config

    def _init_decoder(self, model_name):

        decoder_config = AutoConfig.from_pretrained(model_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        kwargs_decoder = {"config": decoder_config}
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs_decoder
        )

        decoder_base_model_prefix = self.decoder.base_model_prefix
        PreTrainedModel._tie_encoder_decoder_weights(
            self.model.encoder,
            self.decoder._modules[decoder_base_model_prefix],
            self.decoder.base_model_prefix,
        )

    def model_forward(self, batch):

        input_sentences, target_sentences = batch

        input_tokenized = self.tokenizer(
            list(input_sentences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("truncate", 120),
        )

        target_tokenized = self.tokenizer(
            list(target_sentences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("truncate", 120),
        )
        target_tokenized = batch_to_device(target_tokenized, target_device=self.device)
        encoded_embedding = self.model.encode_tokenized_input(
            input_tokenized, device=self.device
        )

        # Prepare input and output
        target_length = target_tokenized["input_ids"].shape[1]
        decoder_input_ids = target_tokenized["input_ids"].clone()[
            :, : target_length - 1
        ]
        label_ids = target_tokenized["input_ids"][:, 1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=None,
            encoder_hidden_states=encoded_embedding[
                :, None
            ],  # (bsz, hdim) -> (bsz, 1, hdim)
            encoder_attention_mask=input_tokenized["attention_mask"][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False,
        )

        # Calculate loss
        lm_logits = decoder_outputs[0]

        loss = self.loss_fn(
            lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1)
        )
        return loss

    def training_step(self, batch, batch_idx=0):

        loss = self.model_forward(batch)

        self.log(
            "train_loss",
            loss,
            batch_size=len(batch[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.get("lr", 1e-3))
        lr_warmup_steps = self.config.get("lr_warm_up_steps", 10000)

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["encoder_state_dict"] = self.model.state_dict()


class MLMFineTuneEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        pretrained_encoder_name: str,
        config: Dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = encoder
        self._init_lm_head(pretrained_encoder_name)
        self.loss_fn = nn.CrossEntropyLoss()
        self.config = config

    def _init_lm_head(self, model_name):

        config = AutoConfig.from_pretrained(model_name)
        self.lm_head = BertOnlyMLMHead(config)

    def model_forward(self, batch):

        batch = batch_to_device(batch, target_device=self.device)
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        last_hidden_state = self.model.encoder(**model_input).last_hidden_state
        prediction_scores = self.lm_head(last_hidden_state)

        masked_lm_loss = self.loss_fn(
            prediction_scores.view(-1, prediction_scores.size(2)), labels.view(-1)
        )

        return masked_lm_loss

    def training_step(self, batch, batch_idx=0):

        loss = self.model_forward(batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.get("lr", 1e-3))
        lr_warmup_steps = self.config.get("lr_warm_up_steps", 10000)

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["encoder_state_dict"] = self.model.state_dict()


def _test_model():
    from models.classifier_head import ClassifierHead

    encoder = PretrainedSentenceEncoder("bert-base-uncased")
    head = ClassifierHead(encoder.hidden_size, num_classes=3)

    from data_utils.nli_data import AllNliTextDataModule

    model = ClassificationFinetuneEncoder(
        encoder=encoder, prediction_head=head, num_classes=3
    )
    model.to("cuda")
    data_loader = AllNliTextDataModule("snli", batch_size=16).val_dataloader()

    for batch in data_loader:
        model.training_step(batch=batch, batch_idx=0)
        break


def _test_triplet_model():
    from losses.multiple_negative_ranking_loss import MultipleNegativesRankingLoss

    encoder = PretrainedSentenceEncoder("bert-base-uncased")
    loss = MultipleNegativesRankingLoss()

    from data_utils.nli_data import AllNliTripletTextDataModule

    model = NPairContrastiveFinetuneEncoder(encoder=encoder, contrast_loss=loss)
    model.to("cuda")
    data_loader = AllNliTripletTextDataModule("snli", batch_size=16).train_dataloader()

    for batch in data_loader:
        model.training_step(batch=batch, batch_idx=0)
        break


def _test_denoise_model():

    encoder = PretrainedSentenceEncoder("bert-base-uncased")

    from data_utils.wiki_data import WikiTextDataModule

    model = DenoisingAEFinetuneEncoder(encoder, "bert-base-uncased")
    model.to("cuda")
    data_loader = WikiTextDataModule("snli", batch_size=16).train_dataloader()

    for batch in data_loader:
        model.training_step(batch=batch, batch_idx=0)
        break


def _test_mlm_model():

    encoder = PretrainedSentenceEncoder("bert-base-uncased")

    from data_utils.wiki_data import MLMWikiTextDataModule

    model = MLMFineTuneEncoder(encoder, "bert-base-uncased")
    model.to("cuda")
    data_loader = MLMWikiTextDataModule(
        "snli", "bert-base-uncased", 128, batch_size=16
    ).train_dataloader()

    for batch in data_loader:
        model.training_step(batch=batch, batch_idx=0)
        break


if __name__ == "__main__":
    _test_mlm_model()
