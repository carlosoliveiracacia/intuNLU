import pytorch_lightning as pl
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import T5ForConditionalGeneration


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
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
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

        src_ids, src_mask, labels = batch

        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        # Run the model and get the logits
        outputs = self(
            input_ids=src_ids,
            attention_mask=src_mask,
            labels=labels
        )

        loss = outputs.loss

        return {'loss': loss}

    def training_step_end(self, batch_parts):
        train_loss = batch_parts['loss'].sum()
        return {'loss': train_loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        print(f'Train loss: {avg_loss} (epoch {self.current_epoch})')
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):

        src_ids, src_mask, labels = batch

        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=src_ids,
            attention_mask=src_mask,
            labels=labels
        )

        loss = outputs.loss

        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        print(f'Val loss: {avg_loss} (epoch {self.current_epoch})')
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)

# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer,
            batch_size,
            max_num_samples=None,
            max_length=512
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_num_samples = max_num_samples
        self.max_length = max_length
        self.train, self.valid, self.test = self.load_split_prepare_data()

    def load_split_prepare_data(self):
        dataset = load_dataset('xsum')
        out = {}
        for ds in dataset.keys():
            n = self.max_num_samples if self.max_num_samples is not None else len(dataset[ds]['document'])
            dataset_x = self.tokenizer(
                dataset[ds]['document'][:n],
                max_length=self.max_length,
                padding='longest',
                return_tensors='pt')

            dataset_y = self.tokenizer(
                dataset[ds]['summary'][:n],
                max_length=self.max_length,
                padding='longest',
                return_tensors='pt'
            )
            out[ds] = {
                'input_ids': dataset_x['input_ids'],
                'attention_mask': dataset_x['attention_mask'],
                'labels': dataset_y['input_ids']
            }

        return out['train'], out['validation'], out['test']

    # Load the training, validation  sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])
        train_data = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
            drop_last=True
        )
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.valid['input_ids'], self.valid['attention_mask'], self.valid['labels'])
        val_data = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels'])
        val_data = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
        return val_data
