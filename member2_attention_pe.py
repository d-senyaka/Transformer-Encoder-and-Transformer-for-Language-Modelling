# member2_attention_pe.py
# Member 2 – Positional Encoding + Self-Attention sandbox (with end-to-end check)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from updated_member1_data_pipeline import build_vocab_indexer, build_dataloader

# ===== Constants =====
SEQ_LEN = 20
D_MODEL = 64
D_INTERNAL = 64
BATCH_SIZE = 8


# ===== Data batch helper =====
def get_batch():
    v, idx = build_vocab_indexer()
    loader = build_dataloader("data/lettercounting-train.txt", idx, task="BEFORE", batch_size=BATCH_SIZE)
    xb, yb = next(iter(loader))
    assert xb.shape == (BATCH_SIZE, SEQ_LEN), f"inputs shape {xb.shape}"
    assert yb.shape == (BATCH_SIZE, SEQ_LEN), f"labels shape {yb.shape}"
    print("✅ Batch OK:", xb.shape, yb.shape)
    return xb, yb, len(idx)


# ===== Positional Encoding (sinusoidal or learned) =====
class PositionalEncoding(nn.Module):
    """
    Adds position info to token embeddings.
    - learned=False → sinusoidal (no trainable params, deterministic)
    - learned=True  → learned table (trainable)
    Works with variable L <= max_len (default 20).
    """
    def __init__(self, d_model=64, max_len=20, learned=False):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.learned = learned

        if learned:
            self.pe = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe_table", pe.unsqueeze(0), persistent=False)  # [1,L,d]

    def forward(self, x):
        """
        x: [B, L, d_model]
        returns: x + PE (broadcast-safe)
        """
        B, L, D = x.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"
        assert L <= self.max_len, f"seq len {L} exceeds max_len {self.max_len}"

        if self.learned:
            positions = torch.arange(0, L, device=x.device)   # [L]
            pe = self.pe(positions).unsqueeze(0)               # [1,L,d]
        else:
            pe = self.pe_table[:, :L, :].to(dtype=x.dtype, device=x.device)  # [1,L,d]
        return x + pe


# ===== Self-Attention (single-head) + FFN =====
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
        A = F.softmax(scores, dim=-1)                           # [B,20,20]
        context = A @ V                                          # [B,20,d_model]
        y = x + context
        y = y + self.ff(y)
        return y, A


# ===== Mini-Transformer (Embed → PE → Self-Attention) =====
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=27, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layer = TransformerLayer(d_model)

    def forward(self, x_idx):
        h = self.emb(x_idx)       # [B,20,d_model]
        h = self.pe(h)            # [B,20,d_model]
        h, A = self.layer(h)      # [B,20,d_model], [B,20,20]
        return h, A


# ===== Quick forward (row-sum check) =====
def _quick_forward_test():
    xb, yb, vocab = get_batch()
    model = MiniTransformer(vocab_size=vocab, d_model=D_MODEL)
    y, A = model(xb)
    assert y.shape == (xb.size(0), SEQ_LEN, D_MODEL), y.shape
    assert A.shape == (xb.size(0), SEQ_LEN, SEQ_LEN), A.shape
    row_sums = A[0].sum(dim=-1)
    print("Row sums (should be ~1):", row_sums)
    print("✅ MiniTransformer forward OK:", y.shape, A.shape)


# ===== PE unit tests (sinusoidal + learned) =====
def _pe_unit_tests():
    B, L, D = 4, 20, 64
    x = torch.zeros(B, L, D)

    # Sinusoidal
    pe_sin = PositionalEncoding(d_model=D, max_len=L, learned=False)
    y = pe_sin(x)
    assert y.shape == (B, L, D)
    assert not torch.allclose(y[:, 0, :], y[:, 1, :]), "Position 0 and 1 encs should differ"
    assert y.abs().sum() > 0, "PE should add non-zero signal"
    assert sum(p.numel() for p in pe_sin.parameters()) == 0, "Sinusoidal PE should have no params"

    # Learned
    pe_learn = PositionalEncoding(d_model=D, max_len=L, learned=True)
    y2 = pe_learn(x.requires_grad_(True))
    assert y2.shape == (B, L, D)
    loss = y2.sum()
    loss.backward()
    grad_sum = pe_learn.pe.weight.grad.abs().sum().item()
    assert grad_sum > 0, "Learned PE weights should receive gradient"

    print("✅ PositionalEncoding tests passed (sinusoidal + learned)")


# ===== PE+Attention test (learned PE toggle) =====
def _mini_transformer_with_pe_test(learned=False):
    class _LocalLayer(nn.Module):
        def __init__(self, d_model=64, d_internal=64):
            super().__init__()
            self.Wq = nn.Linear(d_model, d_internal)
            self.Wk = nn.Linear(d_model, d_internal)
            self.Wv = nn.Linear(d_model, d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model)
            )
        def forward(self, x):
            Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
            scores = Q @ K.transpose(-2, -1) / (K.size(-1)**0.5)
            A = F.softmax(scores, dim=-1)         # [B,20,20]
            context = A @ V                        # [B,20,d]
            y = x + context
            y = y + self.ff(y)
            return y, A

    class _Mini(nn.Module):
        def __init__(self, vocab_size=27, d_model=64, learned_pe=False):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model)
            self.pe = PositionalEncoding(d_model=d_model, max_len=20, learned=learned_pe)
            self.layer = _LocalLayer(d_model=d_model, d_internal=64)
        def forward(self, x_idx):
            h = self.emb(x_idx)
            h = self.pe(h)
            h, A = self.layer(h)
            return h, A

    v, idx = build_vocab_indexer()
    loader = build_dataloader("data/lettercounting-train.txt", idx, batch_size=8)
    xb, _ = next(iter(loader))
    model = _Mini(vocab_size=len(idx), d_model=64, learned_pe=learned)
    y, A = model(xb)
    assert y.shape == (xb.size(0), 20, 64)
    assert A.shape == (xb.size(0), 20, 20)
    print(f"✅ MiniTransformer(+PE learned={learned}) OK:", y.shape, A.shape)


# ===== Step 4: End-to-End Mini-Transformer check =====
def _mini_transformer_end_to_end_test():
    """End-to-end integration test: embedding + PE + TransformerLayer (your tested layer)."""
    v, idx = build_vocab_indexer()
    loader = build_dataloader("data/lettercounting-train.txt", idx, batch_size=8)
    xb, _ = next(iter(loader))

    model = MiniTransformer(vocab_size=len(idx), d_model=64)
    model.eval()  # deterministic (dropout off if any is added later)
    y, A = model(xb)
    print("Output:", y.shape, "Attention:", A.shape)
    print("✅ End-to-End Mini-Transformer forward OK")


# ===== Main driver =====
if __name__ == "__main__":
    xb, yb, vocab_size = get_batch()
    print("Vocab size:", vocab_size)  # expect 27
    _quick_forward_test()
    _pe_unit_tests()
    _mini_transformer_with_pe_test(learned=False)
    _mini_transformer_with_pe_test(learned=True)
    _mini_transformer_end_to_end_test()
