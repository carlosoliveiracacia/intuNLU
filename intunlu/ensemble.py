import logging

import torch

from transformers import T5Tokenizer
from rouge_score import rouge_scorer

from intunlu.finetunning import SummarizerModel
from intunlu.generate_ensemble import generate
from intunlu.train import load_data


def load_model(param_list, max_num_samples):
    datasets = load_data(max_num_samples=max_num_samples)

    model_list = []
    for (model_name, random_state, max_input_length) in param_list:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = SummarizerModel(
            model_name=model_name,
            tokenizer=tokenizer,
            learning_rate=2e-5,
            freeze_encoder=False,
            freeze_embeds=False,
            optimizer='Adam',
            max_input_length=max_input_length
        )
        model_list.append(model.load_from_checkpoint(f'summarizer_{random_state}'))

    results = {}
    results['test'] = ensemble(model_list, datasets['test'], max_input_length)


def ensemble(model, dataset, max_input_length):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    results = {
        'reference_summary': [],
        'predicted_summary': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    for i in range(len(dataset['document'])):
        document = model.tokenizer(
            'summarize: ' + dataset['document'][i],
            max_length=max_input_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            pred = generate([model], document, device, max_input_length)
            pred = model.tokenizer.convert_ids_to_tokens(pred[0], skip_special_tokens=True)
            pred = model.tokenizer.convert_tokens_to_string(pred).replace(' . ', '. ')
            if i < 2:
                logging.info('The input document:')
                logging.info(dataset['document'][i])
                logging.info('The reference summary:')
                logging.info(dataset['summary'][i])
                logging.info('The predicted summary:')
                logging.info(pred)
        results['reference_summary'].append(dataset['summary'][i])
        results['predicted_summary'].append(pred)
        scores = scorer.score(results['reference_summary'][0], results['predicted_summary'][0])
        results['rouge1'].append(scores['rouge1'][1])
        results['rouge2'].append(scores['rouge2'][1])
        results['rougeL'].append(scores['rougeL'][2])

    return results


if __name__ == '__main__':
    load_model()
