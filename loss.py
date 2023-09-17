import torch.nn as nn


class PretrainingLoss(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_token_ids, gt_token_ids, mlm_mask, gt_sent_order, pred_sent_order):
        gt_token_ids[~mlm_mask] = -100
        mlm_loss = self.ce(pred_token_ids.view(-1, self.vocab_size), gt_token_ids.view(-1))

        sop_loss = self.ce(pred_sent_order, gt_sent_order)
        return mlm_loss, sop_loss
