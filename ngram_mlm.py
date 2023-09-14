# References
    # https://github.com/graykode/ALBERT-Pytorch/blob/master/utils.py

import torch
import numpy as np
import random

import config
from sentencepiece import load_fast_albert_tokenizer


GRAMS = [1, 2, 3]
VALS = np.array([1 / gram for gram in GRAMS])
PROBS = VALS / sum(VALS)


def _sample_n():
    return np.random.choice(GRAMS, p=PROBS)


def _get_mask_ratio(select_mask):
    return (sum(select_mask) / len(select_mask)).item()


def perform_ngram_mlm(text, tokenizer, select_prob=0.15):
    # no_mask_token_ids
    encoding = tokenizer(text)
    tokens = encoding.tokens()

    select_mask = torch.zeros(size=(len(tokens),), dtype=bool)
    while _get_mask_ratio(select_mask) < select_prob:
        n = _sample_n()
        start_idx = random.sample(
            [
                idx for idx, token in enumerate(tokens)
                if (token not in ["[CLS]", "[SEP]"]) and (not select_mask[idx])
            ],
        k=1,
        )[0]
        if (tokens[start_idx][0] == "▁") and (not select_mask[start_idx - 1]) and (not select_mask[start_idx + n]):
            select_mask[start_idx: start_idx + n] = True

    gt_token_ids = encoding["input_ids"]
    return gt_token_ids, select_mask
text = "English texts for beginners to practice reading and comprehension online and for free. Practicing your comprehension of written English will both improve your vocabulary and understanding of grammar and word order. The texts below are designed to help you develop while giving you an instant evaluation of your progress."
gt_token_ids, select_mask = perform_ngram_mlm(
    text, tokenizer=tokenizer,
)
len(gt_token_ids), select_mask.shape


# "We generate masked inputs for the MLM targets using n-gram masking (Joshi et al., 2019), with the length of each n-gram mask selected randomly. The probability for the length n is given by p(n) = 1=n PN k=1 1=k We set the maximum length of n-gram (i.e., n) to be 3 (i.e., the MLM target can consist of up to a 3-gram of complete words, such as 'White House correspondents')"

# "Given a sequence of tokensX = (x1; x2; : : : ; xn), we select a subset of tokens Y   X by iteratively sampling spans of text until the masking budget (e.g. 15% of X) has been spent. At each iteration, we first sample a span length (number of words) from a geometric distribution `   Geo(p), which is skewed towards shorter spans. We then randomly (uniformly) select the starting point for the span to be masked. We always sample a sequence of complete words (instead of subword tokens) and the starting point must be the beginning of one word."

# "As in BERT, we also mask 15% of the tokens in total: replacing 80% of the masked tokens with [MASK], 10% with random tokens and 10% with the original tokens. However, we perform this replacement at the span level and not for each token individually; i.e. all the tokens in a span are replaced with [MASK]or sampled tokens."