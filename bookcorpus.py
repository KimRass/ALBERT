# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

import os
import torch
from torch.utils.data import Dataset
import numpy as np

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


def _sample_mask(seg, mask_alpha=4, mask_beta=1, max_gram=3, goal_num_predict=77):
    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    # 3-gram implementation
    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype="bool")

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True) # p(n) = 1/n / sigma(1/k)

    cur_len = 0

    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)

        # `mask_alpha` : number of tokens forming group
        # `mask_beta` : number of tokens to be masked in each groups.
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx

        while beg < seg_len and not _is_word_start_token(token=[seg[beg]], word_start_tokens=config.WORD_START_TOKENS):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_word_start_token(token=[seg[beg]], word_start_tokens=config.WORD_START_TOKENS):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    tokens, masked_tokens, masked_pos = [], [], []
    for i in range(seg_len):
        if mask[i] and (seg[i] != '[CLS]' and seg[i] != '[SEP]'):
            masked_tokens.append(seg[i])
            masked_pos.append(i)
            tokens.append('[MASK]')
        else:
            tokens.append(seg[i])
    return masked_tokens, masked_pos, tokens


class BookCorpusForALBERT(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
        mode="full_sentences",
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mode = mode

        self.unk_id = tokenizer.unk_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.lines = parse(epubtxt_dir, with_document=True)

    def _to_bert_input(self, token_ids):
        # Add "[CLS]" and the first "[SEP]" tokens.
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        token_ids += [self.pad_id] * (self.seq_len - len(token_ids)) # Pad.
        return torch.as_tensor(token_ids)

    def _pad(self, x):
        x = [-1] + x + [-1]
        x += [-1] * (self.seq_len - len(x))
        return torch.as_tensor(x)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # print(f"init_idx: {idx}")
        gt_token_ids = list()
        is_start_ls = list()
        temp = list()
        prev_doc = self.lines[idx][0]
        while True:
            if idx >= len(self.lines) - 1:
                break
                
            cur_doc, line = self.lines[idx]
            tokens, token_ids = _encode(line, tokenizer=self.tokenizer)
            is_start = [1 if _is_word_start_token(token=token, word_start_tokens=config.WORD_START_TOKENS) else 0 for token in tokens]
            if len(gt_token_ids) + len(token_ids) >= self.seq_len - 2:
                break

            if prev_doc != cur_doc:
                gt_token_ids.append(self.sep_id)

            gt_token_ids.extend(token_ids)
            is_start_ls.extend(is_start)
            temp.extend(tokens)
            prev_doc = cur_doc
            idx += 1

        gt_token_ids = self._to_bert_input(gt_token_ids)
        is_start_ls = self._pad(is_start_ls)
        seg_ids = _token_ids_to_segment_ids(token_ids=gt_token_ids, sep_id=self.sep_id)
        # masked_token_ids, masked_pos, tokens = _sample_mask(temp)
        # print(masked_token_ids)
        # print(masked_pos)
        # print(tokens)
        return gt_token_ids, seg_ids, is_start_ls

# "We always limit the maximum input length to 512, and randomly generate input sequences
# shorter than 512 with a probability of 10%."

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