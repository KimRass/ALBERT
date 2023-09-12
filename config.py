import torch
from pathlib import Path

### Data
# "Like BERT, we use a vocabulary size of 30,000, tokenized using SentencePiece."
VOCAB_SIZE = 30_000
VOCAB_DIR = Path(__file__).parent/"bookcorpus_vocab"
MAX_LEN = 512

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
SELECT_PROB = 0.15
MASK_PROB = 0.8
RANDOMIZE_PROB = 0.1

### Optimizer
# "All the model updates use a LAMB optimizer with learning rate 0.00176."
LR = 0.00176

### Training
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
N_WORKERS = 4
# "All the model updates use a batch size of 4096."
# "We train all models for 125,000 steps."
DEFAULT_BATCH_SIZE = 4096
DEFAULT_N_STEPS = 125_000
CKPT_DIR = Path(__file__).parent/"pretraining_checkpoints"
N_CKPT_SAMPLES = 100_000
