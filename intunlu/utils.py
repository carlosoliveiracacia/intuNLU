import logging
import random

from datasets import load_dataset


def load_data(f_train_samples=None, random_state=None, add_header=False):
    # load datasets and trim them down as needed
    datasets = {}
    for ds in ['train', 'validation', 'test']:
        dataset = load_dataset('xsum', split=ds)
        dataset = list(zip(dataset['document'], dataset['summary']))
        logging.info(f'{ds} dataset: initial # of samples is {len(dataset)}.')
        if ((f_train_samples is not None) and (ds=='train')):
            random.seed(random_state)
            dataset = random.sample(dataset, int(f_train_samples * len(dataset)))
        logging.info(f'{ds} dataset: will effectively use {len(dataset)} samples.')

        if add_header:
            datasets[ds] = {
                'document': ['summarize: ' + e[0].replace('\n', ' ') for e in dataset],
                'summary': [e[1] for e in dataset]
            }
        else:
            datasets[ds] = {
                'document': [e[0].replace('\n', ' ') for e in dataset],
                'summary': [e[1] for e in dataset]
            }

    return datasets


def setup_logger(random_state):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  [%(filename)s:%(lineno)s] [%(funcName)20s()] %(message)s",
        handlers=[
            logging.FileHandler(f"{random_state}.log"),
            logging.StreamHandler()
        ]
    )