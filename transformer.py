# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

import torch.nn.functional as F
import math

# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
# ✅
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: 27 (a–z + space)
        :param num_positions: 20
        :param d_model: model width (e.g., 64)
        :param d_internal: attention inner dim (keys/queries), e.g., 64
        :param num_classes: 3
        :param num_layers: stack depth (start with 1; you can try 2 later)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Embedding + positional encoding (unbatched flow in this file)
        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model, num_positions=num_positions, batched=False)

        # Stack of attention layers (your TransformerLayer from below)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])

        # Per-position classifier head → 3 classes
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, indices: torch.LongTensor):
        """
        indices: [L] = [20] long tensor of character ids
        returns:
          - log_probs: [L, 3] (log-softmax over classes at each position)
          - attn_maps: list of [L, L] attention matrices (one per layer)
        """
        # [L] → [L, d_model]
        x = self.char_emb(indices)
        # add positional encodings (unbatched path)
        x = self.posenc(x)  # [L, d_model]

        # Our TransformerLayer expects [B,L,d], so add a batch dim
        x = x.unsqueeze(0)  # [1, L, d_model]
        attn_maps = []
        for layer in self.layers:
            x, A = layer(x)            # x: [1, L, d], A: [1, L, L]
            attn_maps.append(A.squeeze(0))  # store [L, L]

        x = x.squeeze(0)               # [L, d_model]
        logits = self.out(x)           # [L, 3]
        log_probs = F.log_softmax(logits, dim=1)  # [L, 3]
        return log_probs, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
# ✅
class TransformerLayer(nn.Module):
    """
    Single self-attention + feedforward block (simplified, single-head).
    Returns (output, attention_weights).
    """
    def __init__(self, d_model=64, d_internal=64, dropout=0.1):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_internal)
        self.Wk = nn.Linear(d_model, d_internal)
        self.Wv = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_internal)

        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: [B,L,d_model]
        mask: optional causal mask [L,L] or [B,L,L] with -inf where disallowed
        """
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B,L,L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        A = F.softmax(scores, dim=-1)              # [B,L,L]
        A = self.dropout(A)
        context = torch.matmul(A, V)               # [B,L,d_model]

        # Residual + normalization
        x = self.norm1(x + context)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, A

def make_causal_mask(L=20, device="cpu"):
    """Upper-triangular mask for causal attention (1=allowed,0=blocked)."""
    mask = torch.tril(torch.ones(L, L, device=device))
    return mask


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
# ✅
def train_classifier(args, train, dev):
    """
    Minimal trainer for Part 1 that:
      - builds the model
      - trains with NLLLoss on unbatched examples
      - returns the trained model (ready for decode())
    """
    # Hyperparameters (feel free to tune later)
    d_model = 64
    d_internal = 64
    num_layers = 1           # try 2 later for Q2 plots/accuracy
    num_positions = 20
    num_classes = 3
    vocab_size = 27

    model = Transformer(
        vocab_size=vocab_size,
        num_positions=num_positions,
        d_model=d_model,
        d_internal=d_internal,
        num_classes=num_classes,
        num_layers=num_layers
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()

    # A few fast epochs are enough to verify end-to-end training
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        # shuffle example indices
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)
        epoch_loss = 0.0

        for ex_i in ex_idxs:
            ex = train[ex_i]  # LetterCountingExample
            log_probs, _ = model(ex.input_tensor)              # [L,3]
            loss = loss_fcn(log_probs, ex.output_tensor)       # targets: [L]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # quick progress
        print(f"[epoch {epoch+1}/{num_epochs}] loss={epoch_loss:.4f}")

    model.eval()
    return model









####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
