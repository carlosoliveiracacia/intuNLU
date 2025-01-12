import logging
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration


class SummarizerModel(pl.LightningModule):
    # Instantiate the model
    def __init__(
            self,
            model_name,
            tokenizer,
            learning_rate=2e-5,
            freeze_encoder=False,
            freeze_embeds=False,
            optimizer='Adam'
    ):
        super().__init__()
        self.save_hyperparameters()
        if 't5' in model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif 'bart' in model_name:
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        if freeze_encoder:
            self._freeze_params(self.model.get_encoder())

        if freeze_embeds:
            self._freeze_embeds()

    @staticmethod
    def _freeze_params(model):
        ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
          adapted from finetune.py '''
        for layer in model.parameters():
            layer.requires_grade = False

    def _freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f'value of `optimizer` = {self.optimizer} not valid.')
        return optimizer

    def training_step(self, batch, batch_idx):

        documents, summaries = batch

        documents = self.tokenizer(
            documents,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        summaries = self.tokenizer(
            summaries,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

        src_ids, src_mask, labels = documents['input_ids'], documents['attention_mask'], summaries['input_ids']

        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = self(
            input_ids=src_ids.to(device),
            attention_mask=src_mask.to(device),
            labels=labels.to(device)
        )

        loss = outputs.loss

        return {'loss': loss}

    def training_step_end(self, batch_parts):
        train_loss = batch_parts['loss'].sum()
        return {'loss': train_loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        logging.info(f'Train loss: {avg_loss} (epoch {self.current_epoch})')
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):

        documents, summaries = batch

        documents = self.tokenizer(
            documents,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        summaries = self.tokenizer(
            summaries,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

        src_ids, src_mask, labels = documents['input_ids'], documents['attention_mask'], summaries['input_ids']

        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = self(
            input_ids=src_ids.to(device),
            attention_mask=src_mask.to(device),
            labels=labels.to(device)
        )

        loss = outputs.loss

        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        logging.info(f'Val loss: {avg_loss} (epoch {self.current_epoch})')
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)


class SummaryDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['document'])

    def __getitem__(self, idx):
        return self.data['document'][idx], self.data['summary'][idx]


# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset,
            valid_dataset,
            tokenizer,
            batch_size
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train = train_dataset
        self.valid = valid_dataset

    # Load the training, validation  sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = SummaryDataset(self.train)
        train_data = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
            drop_last=True
        )
        return train_data

    def val_dataloader(self):
        dataset = SummaryDataset(self.valid)
        val_data = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
        return val_data
