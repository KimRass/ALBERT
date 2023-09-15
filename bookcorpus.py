# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

import os
import torch
from torch.utils.data import Dataset

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


class BookCorpusForALBERT(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
        ngram_mlm,
        mode="full_sentences",
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.ngram_mlm = ngram_mlm
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

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        print(f"init_idx: {idx}")
        gt_token_ids = list()
        new_tokens = list()
        prev_doc = self.lines[idx][0]
        while True:
            if idx >= len(self.lines) - 1:
                break
                
            cur_doc, line = self.lines[idx]
            tokens, token_ids = _encode(line, tokenizer=self.tokenizer)
            # is_start = [token[0] == "▁" for token in tokens]
            if len(gt_token_ids) + len(token_ids) >= self.seq_len - 2:
                break

            if prev_doc != cur_doc:
                gt_token_ids.append(self.sep_id)

            gt_token_ids.extend(token_ids)
            new_tokens.extend(tokens)
            prev_doc = cur_doc
            idx += 1
            # print(idx, len(new_tokens))

        gt_token_ids = self._to_bert_input(gt_token_ids)
        seg_ids = _token_ids_to_segment_ids(token_ids=gt_token_ids, sep_id=self.sep_id)
        # masked_token_ids, mlm_mask = self.ngram_mlm(tokens=tokens, gt_token_ids=gt_token_ids)
        # return gt_token_ids, masked_token_ids, mlm_mask, seg_ids
        return gt_token_ids, seg_ids

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