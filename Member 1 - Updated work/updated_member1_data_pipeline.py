# updated_member1_data_pipeline.py
# Member 1 — Data Pipeline + Shape Verification for the letter-counting Transformer
# Compatible with letter_counting.py framework

import torch
from torch.utils.data import Dataset, DataLoader
from utils import Indexer
import numpy as np
import os


# ------------------------------------------------------------------
# 1.  Vocabulary and indexer setup (must match letter_counting.py)
# ------------------------------------------------------------------
def build_vocab_indexer():
    vocab = [chr(ord('a') + i) for i in range(26)] + [' ']
    vocab_index = Indexer()
    for c in vocab:
        vocab_index.add_and_get_index(c)
    return vocab, vocab_index


# ------------------------------------------------------------------
# 2.  Label generation (mirrors get_letter_count_output)
# ------------------------------------------------------------------
def get_labels(seq: str, count_only_previous=True):
    labels = np.zeros(len(seq), dtype=int)
    for i in range(len(seq)):
        if count_only_previous:
            labels[i] = min(2, len([c for c in seq[:i] if c == seq[i]]))
        else:
            labels[i] = min(2, len([c for c in seq if c == seq[i]]) - 1)
    return labels


# ------------------------------------------------------------------
# 3.  String → index converter
# ------------------------------------------------------------------
def line_to_indices(s, vocab_index):
    """Convert a 20-char string into a list of integer indices using vocab_index."""
    return [vocab_index.index_of(ch) for ch in s]


# ------------------------------------------------------------------
# 4.  PyTorch dataset wrapper
# ------------------------------------------------------------------
class LetterCountingTorchDataset(Dataset):
    """Custom Dataset for the letter-counting task."""
    def __init__(self, lines, labels, vocab_index):
        self.lines = lines
        self.labels = labels
        self.vocab_index = vocab_index

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        x = torch.tensor(line_to_indices(self.lines[idx], self.vocab_index), dtype=torch.long)  # [L]
        y = torch.tensor(self.labels[idx], dtype=torch.long)                                     # [L]
        return x, y


# ------------------------------------------------------------------
# 5.  Collate + DataLoader builder
# ------------------------------------------------------------------
def collate_fn(batch):
    """Combine examples into batch tensors."""
    xs = torch.stack([x for x, _ in batch])  # [B, L]
    ys = torch.stack([y for _, y in batch])  # [B, L]
    return xs, ys


def build_dataloader(txt_path, vocab_index, task="BEFORE", batch_size=32, shuffle=True):
    """Build DataLoader for train/dev text files."""
    count_only_previous = task == "BEFORE"

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n\r") for ln in f]

    # sanity checks
    assert all(len(s) == 20 for s in lines), "Each input line must be 20 characters long."
    assert len(vocab_index) == 27, "Vocab size mismatch (expected 27)."

    labels = [get_labels(ln, count_only_previous) for ln in lines]
    ds = LetterCountingTorchDataset(lines, labels, vocab_index)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader


# ------------------------------------------------------------------
# 6.  Single-example converter for debugging / demo
# ------------------------------------------------------------------
def convert_single_example(s, vocab_index, task="BEFORE"):
    """Convert a single string into input/label tensors of shape [1, L]."""
    y = get_labels(s, count_only_previous=(task == "BEFORE"))
    x = torch.tensor(line_to_indices(s, vocab_index), dtype=torch.long).unsqueeze(0)  # [1, L]
    y = torch.tensor(y, dtype=torch.long).unsqueeze(0)                                # [1, L]
    return x, y


# ------------------------------------------------------------------
# 7.  Interface / shape checker (use with Member 3’s Transformer)
# ------------------------------------------------------------------
def check_model_interface(model, loader):
    """Verify that model outputs log_probs with shape [B, L, 3]."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        out = model(inputs)
        if isinstance(out, tuple):
            log_probs, attn = out
        else:
            log_probs, attn = out, None

    assert log_probs.shape[:2] == labels.shape, \
        f"Shape mismatch: log_probs {log_probs.shape}, labels {labels.shape}"

    print("✅  Model interface OK")
    print(f"Inputs {inputs.shape}, Labels {labels.shape}, LogProbs {log_probs.shape}")
    if attn is not None:
        print(f"Attention tensor/list detected: {type(attn)}")


# ------------------------------------------------------------------
# 8.  Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    vocab, vocab_index = build_vocab_indexer()
    train_path = "data/lettercounting-train.txt"

    if not os.path.exists(train_path):
        print("⚠️  Please place the train/dev files under data/")
        exit()

    for task in ["BEFORE", "BEFOREAFTER"]:
        loader = build_dataloader(train_path, vocab_index, task=task, batch_size=8)
        xb, yb = next(iter(loader))
        print(f"{task} → Inputs {xb.shape}, Labels {yb.shape}")

    # Dummy model to verify interface
    class Dummy(torch.nn.Module):
        def __init__(self, vocab_size=27, d_model=64, n_classes=3):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab_size, d_model)
            self.fc = torch.nn.Linear(d_model, n_classes)
        def forward(self, x):
            h = self.emb(x)
            logits = self.fc(h)
            return torch.nn.functional.log_softmax(logits, dim=-1), None

    dummy = Dummy()
    check_model_interface(dummy, loader)
