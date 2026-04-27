# member1_data_pipeline.py
# Member 1 — Data Pipeline + Shape Verification for the letter-counting Transformer
# Compatible with letter_counting.py framework

import torch
from torch.utils.data import Dataset, DataLoader
from utils import Indexer, LetterCountingExample
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
# 3.  PyTorch dataset wrapper
# ------------------------------------------------------------------
class LetterCountingTorchDataset(Dataset):
    """Wraps the coursework LetterCountingExample objects for batching."""
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Each LetterCountingExample already stores indexed sequence & gold labels
        return (
            torch.tensor(ex.input_word_indexes, dtype=torch.long),  # [seq_len]
            torch.tensor(ex.output, dtype=torch.long)               # [seq_len]
        )


def collate_fn(batch):
    """Combine examples into batch tensors."""
    inputs = torch.stack([b[0] for b in batch])   # [B, L]
    labels = torch.stack([b[1] for b in batch])   # [B, L]
    return inputs, labels


# ------------------------------------------------------------------
# 4.  Loader builder — integrates exactly with letter_counting.py logic
# ------------------------------------------------------------------
def build_dataloader(txt_path, vocab_index, task="BEFORE", batch_size=32, shuffle=True):
    count_only_previous = task == "BEFORE"
    # read raw lines
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip("\n\r") for ln in f]

    examples = []
    for ln in lines:
        labels = get_labels(ln, count_only_previous)
        examples.append(LetterCountingExample(ln, labels, vocab_index))

    dataset = LetterCountingTorchDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader


# ------------------------------------------------------------------
# 5.  Single-example converter for debugging / demo
# ------------------------------------------------------------------
def convert_single_example(s, vocab_index, task="BEFORE"):
    count_only_previous = task == "BEFORE"
    labels = get_labels(s, count_only_previous)
    ex = LetterCountingExample(s, labels, vocab_index)
    inp = torch.tensor(ex.input_word_indexes, dtype=torch.long).unsqueeze(0)  # [1,L]
    lab = torch.tensor(ex.output, dtype=torch.long).unsqueeze(0)
    return inp, lab


# ------------------------------------------------------------------
# 6.  Interface / shape checker (use with Member 3’s Transformer)
# ------------------------------------------------------------------
def check_model_interface(model, loader):
    model.eval()
    inputs, labels = next(iter(loader))
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
# 7.  Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    vocab, vocab_index = build_vocab_indexer()
    train_path = "data/lettercounting-train.txt"
    if not os.path.exists(train_path):
        print("⚠️  Please place the train/dev files under data/")
        exit()

    loader = build_dataloader(train_path, vocab_index, task="BEFORE", batch_size=8)
    inputs, labels = next(iter(loader))
    print("Sample batch shapes:", inputs.shape, labels.shape)

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
