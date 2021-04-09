import logging
import pickle

import pytorch_lightning as pl
import time
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Tokenizer, BartTokenizer

from intunlu.finetunning import SummaryDataModule, SummarizerModel
from intunlu.evaluate import evaluate
from intunlu.utils import setup_logger, load_data


def train(
        model_name='t5-small',
        batch_size=2,
        n_max_epochs=10,
        random_state=1234,
        f_train_samples=2./3.,
        learning_rate=2e-5,
        run_evaluation=False
):
    setup_logger(random_state)

    add_header = True if 't5' in model_name else False
    datasets = load_data(f_train_samples=f_train_samples, random_state=random_state, add_header=add_header)

    pl.utilities.seed.seed_everything(random_state)

    if 't5' in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif 'bart' in model_name:
        tokenizer = BartTokenizer.from_pretrained(model_name)

    data = SummaryDataModule(
        datasets['train'],
        datasets['validation'],
        tokenizer,
        batch_size
    )

    model = SummarizerModel(
        model_name=model_name,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        freeze_encoder=False,
        freeze_embeds=False,
        optimizer='Adam'
    )

    logger = TensorBoardLogger('logger', f'summarizer_{random_state}')

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
        logger=logger,
        callbacks=[early_stop_callback]
    )

    logging.info('Starting training...')
    s = time.time()
    trainer.fit(model, data)
    logging.info('Done with training.')
    logging.info(f'(Took {time.time() - s} seconds.)')
    trainer.save_checkpoint(f'summarizer_{random_state}')

    if run_evaluation:
        logging.info('Starting evaluation...')
        s = time.time()
        results = {}
        results['valid'] = evaluate(model, datasets['validation'])
        results['test'] = evaluate(model, datasets['test'])
        for ds in results:
            logging.info(f'Metrics for {ds} set:')
            for m in ['rouge1', 'rouge2', 'rougeL']:
                logging.info(f'Average {m} score: {sum(results[ds][m]) / len(results[ds][m])}')
        logging.info('Done with evaluation.')
        logging.info(f'(Took {time.time() - s} seconds.)')
        with open(f'results_{random_state}.pkl', 'wb')as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    train()
