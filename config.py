import torch
from pathlib import Path

### Data
# "Like BERT, we use a vocabulary size of 30,000, tokenized using SentencePiece."
VOCAB_SIZE = 30_522
VOCAB_DIR = Path(__file__).parent/"bookcorpus_vocab"
MAX_LEN = 512
WORD_START_TOKENS = {"""!"#$%&\"()*+,-./:;?@[\\]^_`{|}~"""}

### Architecture
# "Following Devlin et al. (2019), we set the feed-forward/filter size to be $4H$ and the number of
# attention heads to be $H / 64$."
N_LAYERS = 12
HIDDEN_SIZE = 768
N_HEADS = HIDDEN_SIZE // 64
EMBED_SIZE = 128
MLP_SIZE = HIDDEN_SIZE * 4

### Regularization
DROP_PROB = 0.1

### Masked Language Model
# We set the maximum length of n-gram (i.e., $n$) to be 3 (i.e., the MLM target can consist of
# up to a 3-gram of complete words, such as 'White House correspondents')"
NGRAM_SIZES = [1, 2, 3]
# "As in BERT, we also mask 15% of the tokens in total: replacing 80% of the masked tokens with
# `"[MASK]"`, 10% with random tokens and 10% with the original tokens. However, we perform this
# replacement at the span level and not for each token individually; i.e. all the tokens in a span
# are replaced with [MASK]or sampled tokens."
MASK_PROB = 0.15
MASK_TOKEN_PROB = 0.8
RANDOM_TOKEN_PROB = 0.1

### Optimizer
# "All the model updates use a LAMB optimizer with learning rate 0.00176."
LR = 0.00176

### Training
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# "All the model updates use a batch size of 4096."
DEFAULT_BATCH_SIZE = 4096
# "We train all models for 125,000 steps."
DEFAULT_N_STEPS = 125_000
CKPT_DIR = Path(__file__).parent/"pretraining_checkpoints"
N_CKPT_SAMPLES = 100_000
