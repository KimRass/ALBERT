# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

import os
import torch
from torch.utils.data import Dataset
import random

import config
from sentencepiece import parse
from ngram_mlm import NgramMLM
from sentencepiece import load_fast_albert_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _encode(x, tokenizer):
    encoding = tokenizer(
        x,
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    tokens = encoding.tokens()[1: -1]
    token_ids = encoding["input_ids"][1: -1]
    return tokens, token_ids


def _token_ids_to_segment_ids(token_ids, sep_id):
    seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
    is_sep = (token_ids == sep_id)
    if is_sep.sum() == 2:
        first_sep, second_sep = is_sep.nonzero()
        # The positions from right after the first '[SEP]' token and to the second '[SEP]' token
        seg_ids[first_sep + 1: second_sep + 1] = 1
    return seg_ids


def _is_word_start_token(token, word_start_tokens):
    joined = "".join(token)
    if (joined.startswith("▁") or joined.startswith("<")
        or joined in word_start_tokens):
        return True
    else:
        return False


class BookCorpusForALBERT(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.unk_id = tokenizer.unk_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.lines = parse(epubtxt_dir)

    def _pad(self, x):
        x = [-1] + x + [-1]
        x += [-1] * (self.seq_len - len(x))
        return torch.as_tensor(x)

    def _to_bert_input(self, former_token_ids, latter_token_ids):
        token_ids = former_token_ids[: self.seq_len - 2]
        # Add "[CLS]" and the first "[SEP]" tokens.
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        if len(token_ids) >= self.seq_len:
            token_ids = token_ids[: self.seq_len]
        else:
            if len(token_ids) < self.seq_len - 1:
                token_ids += latter_token_ids
                token_ids = token_ids[: self.seq_len - 1]
                token_ids += [self.sep_id] # Add the second "[SEP]" token.
            token_ids += [self.pad_id] * (self.seq_len - len(token_ids)) # Pad.
        return torch.as_tensor(token_ids)

    def __len__(self):
        return len(self.lines) - 1

    def __getitem__(self, idx):
        former_tokens, former_token_ids = _encode(self.lines[idx], tokenizer=self.tokenizer)
        latter_tokens, latter_token_ids = _encode(self.lines[idx + 1], tokenizer=self.tokenizer)
        if random.random() < 0.5:
            former_token_ids, latter_token_ids = latter_token_ids, former_token_ids
            former_tokens, latter_tokens = latter_tokens, former_tokens
            gt_sent_order = torch.as_tensor(1)
        else:
            gt_sent_order = torch.as_tensor(0)
        tokens = former_tokens + latter_tokens
        is_start_token = [1 if _is_word_start_token(token=token, word_start_tokens=config.WORD_START_TOKENS) else 0 for token in tokens]
        is_start_token = self._pad(is_start_token)
        gt_token_ids = self._to_bert_input(
            former_token_ids=former_token_ids, latter_token_ids=latter_token_ids,
        )
        seg_ids = _token_ids_to_segment_ids(token_ids=gt_token_ids, sep_id=self.sep_id)
        return is_start_token, gt_token_ids, seg_ids, gt_sent_order


if __name__ == "__main__":
    tokenizer = load_fast_albert_tokenizer(
        "/Users/jongbeomkim/Desktop/workspace/albert_from_scratch/bookcorpus_vocab"
    )
    global ngram_mlm
    ngram_mlm = NgramMLM(
        vocab_size=100,
        unk_id=tokenizer.unk_token_id,
        cls_id=tokenizer.cls_token_id,
        sep_id=tokenizer.sep_token_id,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        seq_len=512,
        no_mask_token_ids=[0, 1, 2, 3, 4],
    )
    ds = BookCorpusForALBERT(
        epubtxt_dir="/Users/jongbeomkim/Documents/datasets/bookcorpus_subset/epubtxt",
        tokenizer=tokenizer,
        seq_len=512,
    )
    gt_token_ids, seg_ids, masked_token_ids, mlm_mask = ds[0]
    gt_token_ids.shape, seg_ids.shape, masked_token_ids.shape, mlm_mask.shape
    gt_token_ids
    masked_token_ids
    mlm_mask