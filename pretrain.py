# References
    # https://github.com/skyday123/pytorch-lamb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import gc
from pathlib import Path
from time import time
from tqdm.auto import tqdm
import argparse
import matplotlib.pyplot as plt
from pytorch_lamb import Lamb

import config
from utils import get_elapsed_time, print_number_of_parameters
from model import ALBERTForPretraining
from sentencepiece import load_fast_albert_tokenizer
from bookcorpus import BookCorpusForALBERT
from ngram_mlm import NgramMLM
from loss import PretrainingLoss
from evalute import get_mlm_acc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epubtxt_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=8192) # "Batch Size"
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args

def prepare_dl(tokenizer, epubtxt_dir, batch_size):
    ngram_mlm = NgramMLM(
        vocab_size=config.VOCAB_SIZE,
        unk_id=tokenizer.unk_token_id,
        cls_id=tokenizer.cls_token_id,
        sep_id=tokenizer.sep_token_id,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        seq_len=config.MAX_LEN,
        mask_prob=config.MASK_TOKEN_PROB,
        mask_token_prob=config.MASK_TOKEN_PROB,
        random_token_prob=config.RANDOM_TOKEN_PROB,
    )
    train_ds = BookCorpusForALBERT(
        epubtxt_dir=epubtxt_dir,
        tokenizer=tokenizer,
        seq_len=config.MAX_LEN,
        ngram_mlm=ngram_mlm,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    return train_dl


def save_checkpoint(step, model, optim, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "optimizer": optim.state_dict(),
    }
    if config.N_GPUS > 1:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()
    torch.save(ckpt, str(ckpt_path))


def resume(ckpt_path, model, optim):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        if config.N_GPUS > 1:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        prev_ckpt_path = Path(ckpt_path)
        print(f"Resuming from checkpoint '{str(Path(ckpt_path).name)}'...")
    else:
        step = 0
        prev_ckpt_path = Path(".pth")
    return step, prev_ckpt_path


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    # gc.collect()
    # torch.cuda.empty_cache()

    args = get_args()

    print(f"BATCH_SIZE = {args.batch_size}")
    print(f"N_WORKERS = {config.N_WORKERS}")
    print(f"MAX_LEN = {config.MAX_LEN}")
    N_STEPS = (config.DEFAULT_BATCH_SIZE * config.DEFAULT_N_STEPS) // args.batch_size
    N_ACCUM_STEPS = config.DEFAULT_BATCH_SIZE // args.batch_size
    print(f"N_STEPS = {N_STEPS:,}")
    print(f"N_ACCUM_STEPS = {N_ACCUM_STEPS:,}")

    tokenizer = load_fast_albert_tokenizer(vocab_dir=config.VOCAB_DIR)
    train_dl = prepare_dl(
        tokenizer=tokenizer, epubtxt_dir=args.epubtxt_dir, batch_size=args.batch_size,
    )

    model = ALBERTForPretraining( # Smaller than BERT-Base
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_LEN,
        pad_id=tokenizer.pad_token_id,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        mlp_size=config.MLP_SIZE,
    ).to(config.DEVICE)
    if config.N_GPUS > 1:
        model = nn.DataParallel(model)

    optim = Adam(model.parameters(), lr=config.LR)
    # optim = Lamb(model.parameters(), lr=config.LR)

    crit = PretrainingLoss(vocab_size=config.VOCAB_SIZE)

    ### Resume
    step, prev_ckpt_path = resume(ckpt_path=args.ckpt_path, model=model, optim=optim)

    print("Training...")
    start_time = time()
    accum_loss = 0
    accum_acc = 0
    step_cnt = 0
    while step < N_STEPS:
        for gt_token_ids, masked_token_ids, mlm_mask, seg_ids in tqdm(train_dl):
            step += 1

    #         gt_token_ids = gt_token_ids.to(config.DEVICE)
    #         masked_token_ids = masked_token_ids.to(config.DEVICE)
    #         mlm_mask = mlm_mask.to(config.DEVICE)
    #         seg_ids = seg_ids.to(config.DEVICE)

    #         pred_token_ids = model(token_ids=masked_token_ids, seg_ids=seg_ids)
    #         loss = crit(
    #             pred_token_ids=pred_token_ids,
    #             gt_token_ids=gt_token_ids,
    #             mlm_mask=mlm_mask,
    #         )

    #         accum_loss += loss.item()
    #         loss /= N_ACCUM_STEPS
    #         loss.backward()

    #         if step % N_ACCUM_STEPS == 0:
    #             optim.step()
    #             optim.zero_grad()

    #         acc = get_mlm_acc(pred_token_ids=pred_token_ids, gt_token_ids=gt_token_ids)
    #         accum_acc += acc
    #         step_cnt += 1

    #         if (step % (config.N_CKPT_SAMPLES // args.batch_size) == 0) or (step == N_STEPS):
    #             print(f"[ {step:,}/{N_STEPS:,} ][ {get_elapsed_time(start_time)} ]", end="")
    #             print(f"[ MLM loss: {accum_loss / step_cnt:.4f} ]", end="")
    #             print(f"[ MLM acc: {accum_acc / step_cnt:.3f} ]")

    #             start_time = time()
    #             accum_loss = 0
    #             accum_acc = 0
    #             step_cnt = 0

    #             cur_ckpt_path = config.CKPT_DIR/f"bookcorpus_step_{step}.pth"
    #             save_checkpoint(step=step, model=model, optim=optim, ckpt_path=cur_ckpt_path)
    #             if prev_ckpt_path.exists():
    #                 prev_ckpt_path.unlink()
    #             prev_ckpt_path = cur_ckpt_path
    # print("Completed pre-training.")
