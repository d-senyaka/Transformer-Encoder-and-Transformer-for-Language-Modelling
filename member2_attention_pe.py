# member2_attention_pe.py
import torch
from updated_member1_data_pipeline import build_vocab_indexer, build_dataloader

import math
import torch.nn as nn
import torch.nn.functional as F

# Constants for the rest of Member 2
SEQ_LEN = 20
D_MODEL = 64
D_INTERNAL = 64
BATCH_SIZE = 8

def get_batch():
    v, idx = build_vocab_indexer()
    loader = build_dataloader("data/lettercounting-train.txt", idx, task="BEFORE", batch_size=BATCH_SIZE)
    xb, yb = next(iter(loader))
    assert xb.shape == (BATCH_SIZE, SEQ_LEN), f"inputs shape {xb.shape}"
    assert yb.shape == (BATCH_SIZE, SEQ_LEN), f"labels shape {yb.shape}"
    print("✅ Batch OK:", xb.shape, yb.shape)
    return xb, yb, len(idx)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=SEQ_LEN, learned=False):
        super().__init__()
        if learned:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
            self.is_learned = True
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
            self.is_learned = False

    def forward(self, x):
        # x: [B, 20, d_model]
        if self.is_learned:
            return x + self.pe[:, :x.size(1)]
        return x + self.pe[:, :x.size(1)]

class TransformerLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, d_internal=D_INTERNAL):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_internal, bias=True)
        self.Wk = nn.Linear(d_model, d_internal, bias=True)
        self.Wv = nn.Linear(d_model, d_model,    bias=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
        )

    def forward(self, x):
        # x: [B, 20, d_model]
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)  # [B,20,20]
        A = F.softmax(scores, dim=-1)
        context = A @ V  # [B,20,d_model]
        y = x + context
        y = y + self.ff(y)
        return y, A

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=27, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layer = TransformerLayer(d_model)

    def forward(self, x_idx):
        h = self.emb(x_idx)        # [B,20,d_model]
        h = self.pe(h)             # [B,20,d_model]
        h, A = self.layer(h)       # [B,20,d_model], [B,20,20]
        return h, A

def _quick_forward_test():
    xb, yb, vocab = get_batch()
    model = MiniTransformer(vocab_size=vocab, d_model=D_MODEL)
    y, A = model(xb)
    assert y.shape == (xb.size(0), SEQ_LEN, D_MODEL), y.shape
    assert A.shape == (xb.size(0), SEQ_LEN, SEQ_LEN), A.shape
    row_sums = A[0].sum(dim=-1)
    print("Row sums (should be ~1):", row_sums)
    print("✅ MiniTransformer forward OK:", y.shape, A.shape)


if __name__ == "__main__":
    xb, yb, vocab_size = get_batch()
    print("Vocab size:", vocab_size)  # expect 27
    _quick_forward_test()
