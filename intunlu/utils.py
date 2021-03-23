import logging
import random

from datasets import load_dataset


def load_data(max_num_samples=None):
    # load datasets and trim them down as needed
    datasets = {}
    for ds in ['train', 'validation', 'test']:
        dataset = load_dataset('xsum', split=ds)
        N = len(dataset)
        logging.info(f'{ds} dataset: initial # of samples is {N}.')
        if max_num_samples is not None:
            n = min(max_num_samples, N)
        else:
            n = N
        logging.info(f'{ds} dataset: will effectively use {n} samples.')
        dataset = list(zip(dataset['document'], dataset['summary']))
        dataset = random.sample(dataset, n)
        datasets[ds] = {
            'document': ['summarize: ' + e[0].replace('\n', ' ') for e in dataset],
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