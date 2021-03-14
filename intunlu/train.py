import pytorch_lightning as pl
import random
import time
import torch

from datasets import load_dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Tokenizer

from intunlu.finetunning import SummaryDataModule, SummarizerModel


def train(
        model_name='t5-small',
        batch_size=2,
        n_max_epochs=10,
        random_state=1234,
        max_num_samples=None
):

    # load datasets and trim them down as needed
    random.seed(1234) # this is meant to be fixed so we can train on a fixed subset of the original training set
    datasets = {}
    for ds in ['train', 'validation']:
        dataset = load_dataset('xsum', split=ds)
        N = len(dataset)
        print(f'{ds} dataset: initial # of samples is {N}.')
        if max_num_samples is not None:
            n = min(max_num_samples, N)
        else:
            n = N
        print(f'{ds} dataset: will effectively use {n} samples.')
        dataset = list(zip(dataset['document'], dataset['summary']))
        dataset = random.sample(dataset, n)
        datasets[ds] = {
            'document': [e[0] for e in dataset],
            'summary': [e[1] for e in dataset]
        }

    pl.utilities.seed.seed_everything(random_state)

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    data = SummaryDataModule(
        datasets['train'],
        datasets['validation'],
        tokenizer,
        batch_size,
        max_num_samples=max_num_samples
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

    print('Starting training...')
    s = time.time()
    trainer.fit(model, data)
    print('Done with training.')
    print(f'(Took {time.time() - s} seconds.)')
    trainer.save_checkpoint(f'summarizer_{random_state}')

#    model.generate()

if __name__ == '__main__':
    train()