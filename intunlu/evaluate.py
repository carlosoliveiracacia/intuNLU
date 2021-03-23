import logging
import torch

from rouge_score import rouge_scorer

from intunlu.ensemble import EnsembleGenerator


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
    model.model.to(device)
    for i in range(len(dataset['document'])):
        document = model.tokenizer(
            'summarize: ' + dataset['document'][i],
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            pred = model.model.generate(
                input_ids=document['input_ids'].to(device),
                attention_mask=document['attention_mask'].to(device),
                use_cache=True,
                decoder_start_token_id=model.tokenizer.pad_token_id,
                num_beams=1,  # greedy search
                early_stopping=True
            )
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

def evaluate_ensemble(paths, dataset, pool_type='mean'):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    results = {
        'reference_summary': [],
        'predicted_summary': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    model = EnsembleGenerator(paths)

    for i in range(len(dataset['document'])):
        pred = model.generate_greedy_search(dataset['document'][i], pool_type=pool_type)
        if i < 2:
            print('The input document:')
            print(dataset['document'][i])
            print('The reference summary:')
            print(dataset['summary'][i])
            print('The predicted summary:')
            print(pred)
        results['reference_summary'].append(dataset['summary'][i])
        results['predicted_summary'].append(pred)
        scores = scorer.score(results['reference_summary'][0], results['predicted_summary'][0])
        results['rouge1'].append(scores['rouge1'][1])
        results['rouge2'].append(scores['rouge2'][1])
        results['rougeL'].append(scores['rougeL'][2])

    print(f'Metrics:')
    for m in ['rouge1', 'rouge2', 'rougeL']:
        print(f'Average {m} score: {sum(results[m]) / len(results[m])}')
    print('Done with evaluation.')
