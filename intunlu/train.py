import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Tokenizer

from intunlu.finetunning import SummaryDataModule, SummarizerModel


def train(
        model_name='t5-small',
        batch_size=8,
        n_max_epochs=10,
        random_state=1234
):

    pl.utilities.seed.seed_everything(random_state)

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    data = SummaryDataModule(
        tokenizer,
        batch_size,
        max_num_samples=32
    )

    model = SummarizerModel(
        model_name=model_name,
        tokenizer=tokenizer,
        learning_rate=2e-5,
        freeze_encoder=False,
        freeze_embeds=False,
        optimizer='Adam'
    )

    logger = TensorBoardLogger('logger', 'summarizer')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=1,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=n_max_epochs,
        min_epochs=1,
        auto_lr_find=False,
        accelerator='dp' if torch.cuda.device_count() > 1 else None,
        logger = logger,
        callbacks=[early_stop_callback]
    )

    trainer.fit(model, data)
    trainer.save_checkpoint(f'summarizer_{random_state}')

#    model.generate()

if __name__ == '__main__':
    train()