import torch


def token_max_score(tokens, scores):
    return tokens


def token_mean_score(tokens, scores):
    return tokens


def fill_matrix(col_len, min_score, seqs, scores):
    for i in range(len(seqs)):
        while len(seqs[i]) < col_len:
            seqs[i].append("")
            scores[i].append(min_score)
    return seqs, scores


def generate(models, document, device, max_input_length):
    seqs = []
    scores = []
    col_len = 0
    min_score = None
    for model in models:
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
        col_len = max(col_len, len(pred[0]))
        seqs.append(pred[0])

        scores.append([torch.max(v) for v in pred[1]])

        if min_score is None:
            min_score = min(scores[-1])
        else:
            min_score = min(min_score, min(scores[-1])-1.0)
    final_output = []
    seqs, scores = fill_matrix(col_len, min_score, seqs, scores)
    for j in range(len(seqs[0])):
        final_output.append(token_max_score(seqs[:][j], scores[:][j]))

    return torch.LongTensor(final_output[0])
