# References
    # https://nn.labml.ai/transformers/mlm/index.html

import torch

import config
from sentencepiece import load_fast_albert_tokenizer


class NgramMLM(object):
    def __init__(
        self,
        vocab_size,
        mask_id,
        no_mask_token_ids=[],
        select_prob=0.15,
        mask_prob=0.8,
        randomize_prob=0.1,
    ):
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.no_mask_token_ids = no_mask_token_ids
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.randomize_prob = randomize_prob

        if mask_id not in no_mask_token_ids:
            no_mask_token_ids += [mask_id]


    def __call__(self, gt_token_ids):
        masked_token_ids = gt_token_ids.clone()

        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        no_mask_mask = torch.isin(
            masked_token_ids,
            torch.as_tensor(self.no_mask_token_ids, device=masked_token_ids.device),
        )
        rand_tensor.masked_fill_(mask=no_mask_mask, value=1)

        select_mask = (rand_tensor < self.select_prob)


        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        mask_mask = select_mask & (rand_tensor < self.mask_prob)
        # `mask_mask.sum() / select_mask.sum() ~= 0.8`
        masked_token_ids.masked_fill_(mask=mask_mask, value=self.mask_id)

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
        return masked_token_ids, select_mask


if __name__ == "__main__":
    tokenizer = load_bert_tokenizer(config.VOCAB_PATH)
    mlm = NgramMLM(
        vocab_size=config.VOCAB_SIZE,
        mask_id=tokenizer.token_to_id("[MASK]"),
        select_prob=config.SELECT_PROB,
        mask_prob=config.MASK_PROB,
        randomize_prob=config.RANDOMIZE_PROB,
    )

# "We generate masked inputs for the MLM targets using n-gram masking (Joshi et al., 2019), with the length of each n-gram mask selected randomly. The probability for the length n is given by p(n) = 1=n PN k=1 1=k We set the maximum length of n-gram (i.e., n) to be 3 (i.e., the MLM target can consist of up to a 3-gram of complete words, such as “White House correspondents”). All the model updates use a batch size of 4096 and a LAMB optimizer with learning rate 0.00176 (You et al., 2019). We train all models for 125,000 steps unless otherwise specified. Training"