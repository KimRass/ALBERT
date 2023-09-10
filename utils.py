import torch
from time import time
from datetime import timedelta


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))

def print_number_of_parameters(model):
    print(f"""{sum([p.numel() for p in model.parameters()]):,}""")


def _token_ids_to_segment_ids(token_ids, sep_id):
    seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
    is_sep = (token_ids == sep_id)
    if is_sep.sum() == 2:
        first_sep, second_sep = is_sep.nonzero()
        # The positions from right after the first '[SEP]' token and to the second '[SEP]' token
        seg_ids[first_sep + 1: second_sep + 1] = 1
    return seg_ids
