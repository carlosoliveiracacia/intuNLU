import pickle
import pytorch_lightning as pl
import random
import time
import torch

from datasets import load_dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from rouge_score import rouge_scorer
from transformers import T5Tokenizer

from intunlu.finetunning import SummaryDataModule, SummarizerModel


def train(
        model_name='t5-small',
        batch_size=2,
        n_max_epochs=10,
        random_state=1234,
        max_num_samples=2000
):
    random.seed(1234)  # this is meant to be fixed so we can train on a fixed subset of the original training set
    datasets = load_data(max_num_samples=max_num_samples)

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

    print('Starting evaluation...')
    s = time.time()
    results = {}
    results['valid'] = evaluate(model, datasets['validation'])
    results['test'] = evaluate(model, datasets['test'])
    for ds in results:
        print(f'Metrics for {ds} set:')
        for m in ['rouge1', 'rouge2', 'rougeL']:
            print(f'Average {m} score: {sum(results[ds][m]) / len(results[ds][m])}')
    print('Done with evaluation.')
    print(f'(Took {time.time() - s} seconds.)')
    with open(f'results_{random_state}.pkl', 'wb')as f:
        pickle.dump(results, f)


def load_data(max_num_samples=None):

    # load datasets and trim them down as needed
    datasets = {}
    for ds in ['train', 'validation', 'test']:
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
            'document': ['summarize: ' + e[0] for e in dataset],
            'summary': [e[1] for e in dataset]
        }

    return datasets

def evaluate(model, dataset):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    results = {
        'reference_summary': [],
        'predicted_summary': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(len(dataset['document'])):
        document = model.tokenizer(
            'summarize: ' + dataset['document'][i],
            max_length=512,
            padding='longest',
            return_tensors='pt'
        )
        with torch.no_grad():
            pred = model.model.generate(
                input_ids=document['input_ids'].to(device),
                attention_mask=document['attention_mask'].to(device),
                use_cache=True,
                decoder_start_token_id=model.tokenizer.pad_token_id,
                num_beams=1,  # greedy search
                max_length=512,
                early_stopping=True
            )
            pred = model.tokenizer.convert_ids_to_tokens(pred[0], skip_special_tokens=True)
            pred = model.tokenizer.convert_tokens_to_string(pred).replace(' . ', '. ')
            if i < 2:
                print('The input document:')
                print(dataset['document'][i])
                print('The reference summary:')
                print(dataset['summary'][i])
                print('The predicted summary:')
                print(pred)
                print()
        results['reference_summary'].append(dataset['summary'][i])
        results['predicted_summary'].append(pred)
        scores = scorer.score(results['reference_summary'][0], results['predicted_summary'][0])
        results['rouge1'].append(scores['rouge1'][1])
        results['rouge2'].append(scores['rouge2'][1])
        results['rougeL'].append(scores['rougeL'][2])

    return results

if __name__ == '__main__':
    train()