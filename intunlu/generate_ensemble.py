import logging

import torch
import numpy as np


def token_max_score(tokens, scores):
    return tokens[scores.argmax()]


def fill_matrix(col_len, min_score, seqs, scores):
    for i in range(len(seqs)):
        while len(seqs[i]) < col_len:
            seqs[i].append("")
            scores[i].append(min_score)
    return seqs, scores


def generate(model_list, document, device, max_input_length, ensemble='max'):
    logging.info("Start generate...")
    seqs = []
    scores = []
    col_len = 0
    min_score = None
    for model in model_list:
        pred = model.model.generate(
            input_ids=document['input_ids'].to(device),
            attention_mask=document['attention_mask'].to(device),
            use_cache=True,
            decoder_start_token_id=model.tokenizer.pad_token_id,
            num_beams=1,  # greedy search
            max_length=max_input_length,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        col_len = max(col_len, len(pred[0][0]))
        seqs.append([v for v in pred[0][0]])
        seqs.append([v for v in pred[0][0]])

        # by default use max
        if ensemble == 'mean':
            scores.append([torch.mean(v) for v in pred[1]])
            scores.append([torch.mean(v) for v in pred[1]])
        else:
            scores.append([torch.max(v) for v in pred[1]])
            scores.append([torch.max(v) for v in pred[1]])

        if min_score is None:
            min_score = min(scores[-1])
        else:
            min_score = min(min_score, min(scores[-1]) - 1.0)
    final_output = []
    seqs, scores = fill_matrix(col_len, min_score, seqs, scores)

    for j in range(1, col_len):
        final_output.append(token_max_score(np.array(seqs)[:, j], np.array(scores)[:, j-1]))

    logging.info("End generate...")
    return torch.LongTensor([final_output])
