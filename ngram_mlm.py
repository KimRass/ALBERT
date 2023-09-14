# References
    # https://github.com/graykode/ALBERT-Pytorch/blob/master/utils.py

import torch
import numpy as np
import random


class NgramMLM(object):
    def __init__(
        self,
        vocab_size,
        unk_id,
        cls_id,
        sep_id,
        pad_id,
        mask_id,
        seq_len,
        no_mask_token_ids=set(),
        ngram_sizes=[1, 2, 3],
        select_prob=0.15,
        mask_prob=0.8,
        randomize_prob=0.1,
    ):
        self.vocab_size = vocab_size
        self.unk_id = unk_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.seq_len = seq_len
        self.no_mask_token_ids = no_mask_token_ids
        self.ngram_sizes = ngram_sizes
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.randomize_prob = randomize_prob

        no_mask_token_ids |= {unk_id, cls_id, sep_id, pad_id, mask_id}

        recips = np.array([1 / gram for gram in ngram_sizes])
        self.probs = recips / sum(recips)

        self.false = torch.zeros(size=(1,), dtype=bool)

    def _sample_ngram_size(self):
        return np.random.choice(self.ngram_sizes, p=self.probs)

    def _get_mask_ratio(self, select_mask):
        return (sum(select_mask) / len(select_mask)).item()

    def _pad(self, select_mask):
        # select_mask = torch.zeros(size=(128,), dtype=bool)
        new_select_mask = [False] + select_mask.tolist() + [False]
        new_select_mask += [False] * (self.seq_len - len(new_select_mask))
        return torch.as_tensor(new_select_mask)

    def _get_select_mask(self, tokens):
        select_mask = torch.zeros(size=(len(tokens),), dtype=bool)
        while self._get_mask_ratio(select_mask) < self.select_prob:
            ngram_size = self._sample_ngram_size()
            start_idx = random.sample(
                [
                    idx for idx, token in enumerate(tokens)
                    if (token not in self.no_mask_token_ids) and (not select_mask[idx])
                ],
            k=1,
            )[0]
            if all([
                tokens[start_idx][0] == "▁",
                not select_mask[start_idx - 1],
                not select_mask[min(len(select_mask) - 1, start_idx + ngram_size)],
            ]):
                select_mask[start_idx: start_idx + ngram_size] = True

        select_mask = self._pad(select_mask)
        return select_mask

    def _replace_some_tokens(self, gt_token_ids, select_mask):
        masked_token_ids = gt_token_ids.clone()

        # "If the $i$-th token is chosen, we replace the $i$-th token with (1) the [MASK] token
        # 80% of the time."
        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        mask_mask = select_mask & (rand_tensor < self.mask_prob)
        # `mask_mask.sum() / select_mask.sum() ~= 0.8`
        masked_token_ids.masked_fill_(mask=mask_mask, value=self.mask_id)

        # "(2) a random token 10% of the time
        # (3) the unchanged $i$-th token 10% of the time."
        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        randomize_mask = select_mask &\
            (rand_tensor >= self.mask_prob) &\
            (rand_tensor < (self.mask_prob + self.randomize_prob))
        # `randomize_mask.sum() / select_mask.sum() ~= 0.1`
        random_token_ids = torch.randint(
            high=self.vocab_size,
            size=torch.Size((randomize_mask.sum(),)),
            device=masked_token_ids.device,
        )
        masked_token_ids[randomize_mask.nonzero(as_tuple=True)] = random_token_ids
        return masked_token_ids

    def __call__(self, tokens, gt_token_ids):
        select_mask = self._get_select_mask(tokens)
        masked_token_ids = self._replace_some_tokens(
            gt_token_ids=gt_token_ids, select_mask=select_mask,
        )
        return masked_token_ids, select_mask


if __name__ == "__main__":
    text = "English texts for beginners to practice reading and comprehension online and for free. Practicing your comprehension of written English will both improve your vocabulary and understanding of grammar and word order. The texts below are designed to help you develop while giving you an instant evaluation of your progress."
    gt_token_ids, select_mask = perform_ngram_mlm(
        text, tokenizer=tokenizer,
    )
    len(gt_token_ids), select_mask.shape


    # "We generate masked inputs for the MLM targets using n-gram masking (Joshi et al., 2019), with the length of each n-gram mask selected randomly. The probability for the length n is given by p(n) = 1=n PN k=1 1=k We set the maximum length of n-gram (i.e., n) to be 3 (i.e., the MLM target can consist of up to a 3-gram of complete words, such as 'White House correspondents')"

    # "Given a sequence of tokensX = (x1; x2; : : : ; xn), we select a subset of tokens Y   X by iteratively sampling spans of text until the masking budget (e.g. 15% of X) has been spent. At each iteration, we first sample a span length (number of words) from a geometric distribution `   Geo(p), which is skewed towards shorter spans. We then randomly (uniformly) select the starting point for the span to be masked. We always sample a sequence of complete words (instead of subword tokens) and the starting point must be the beginning of one word."

    # "As in BERT, we also mask 15% of the tokens in total: replacing 80% of the masked tokens with [MASK], 10% with random tokens and 10% with the original tokens. However, we perform this replacement at the span level and not for each token individually; i.e. all the tokens in a span are replaced with [MASK]or sampled tokens."