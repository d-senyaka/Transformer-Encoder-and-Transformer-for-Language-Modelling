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
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        # Embedding for characters
        self.embed = nn.Embedding(vocab_size, d_model)
        # Positional encoding module (supports batched or unbatched in forward)
        self.pos_enc = PositionalEncoding(d_model, num_positions)
        # Stacked transformer layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        # Final classifier to map d_model -> num_classes at each position
        self.classifier = nn.Linear(d_model, num_classes)
        # We'll output log-probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.d_model = d_model
        self.num_layers = num_layers

    def forward(self, indices):
        """
        :param indices: list of input indices (either LongTensor of shape [seq_len] or [batch, seq_len])
        :return: A tuple of the softmax log probabilities and a list of attention maps used in each layer
                 For unbatched input: returns (seq_len x num_classes, [attn_map_layer0, ...]) where attn_map is seq_len x seq_len
                 For batched input: returns (batch x seq_len x num_classes, [attn_map_layer0]) where attn_map is batch x seq_len x seq_len
        """
        # Detect batching
        batched = (indices.dim() == 2)
        if batched:
            batch_size, seq_len = indices.shape
        else:
            seq_len = indices.shape[0]
            batch_size = None

        # Embedding
        # embed: if unbatched: seq_len x d_model. If batched: batch x seq_len x d_model
        emb = self.embed(indices)
        # Add positional encodings (PositionalEncoding expects batched flag)
        self.pos_enc.batched = batched
        emb = self.pos_enc(emb)  # retains shapes

        # Pass through transformer layers and collect attention maps
        attn_maps = []
        x = emb  # x shape matches emb
        for layer in self.layers:
            x, attn = layer(x)  # layer should preserve shape and return attention map(s)
            attn_maps.append(attn)

        # Final classifier: apply to last dimension (d_model -> num_classes)
        logits = self.classifier(x)  # shape: [seq_len, num_classes] or [batch, seq_len, num_classes]
        log_probs = self.log_softmax(logits)

        # For compatibility with decode(): when unbatched, return [seq_len x num_classes]
        return log_probs, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # Linear maps to produce queries, keys, values
        self.q_lin = nn.Linear(d_model, d_internal)
        self.k_lin = nn.Linear(d_model, d_internal)
        self.v_lin = nn.Linear(d_model, d_internal)

        # Project attention output back to d_model for residual connection
        self.out_lin = nn.Linear(d_internal, d_model)

        # Feed-forward network: two-layer MLP with nonlinearity
        self.ff_hidden = nn.Linear(d_model, d_internal * 2)
        self.ff_out = nn.Linear(d_internal * 2, d_model)
        self.activation = nn.ReLU()

        # For numerical stability in softmax scaling
        self.scale = math.sqrt(d_internal)

    def forward(self, input_vecs):
        """
        :param input_vecs: either [seq_len, d_model] (unbatched) or [batch, seq_len, d_model] (batched)
        :return: (output_vecs, attn_map) where attn_map is
                 - unbatched: [seq_len, seq_len]
                 - batched: [batch, seq_len, seq_len]
        """
        batched = (input_vecs.dim() == 3)
        if not batched:
            # convert to [1, seq_len, d_model] to reuse same code, and squeeze later
            x = input_vecs.unsqueeze(0)  # [1, seq_len, d_model]
        else:
            x = input_vecs  # [batch, seq_len, d_model]

        batch_size, seq_len, d_model = x.shape

        # Compute queries, keys, values
        # Shapes: [batch, seq_len, d_internal]
        Q = self.q_lin(x)
        K = self.k_lin(x)
        V = self.v_lin(x)

        # Compute attention scores: Q @ K^T for each batch
        # We'll compute using bmm after reshaping: for each batch i, attn_scores = Q_i (seq x d) @ K_i^T (d x seq) -> seq x seq
        # Reshape to prepare for bmm: (batch, seq_len, d_internal) and (batch, d_internal, seq_len)
        K_t = K.transpose(1, 2)
        attn_scores = torch.bmm(Q, K_t)  # [batch, seq_len, seq_len]
        # Scale
        attn_scores = attn_scores / self.scale
        # Softmax over keys (last dim)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len]

        # Weighted sum of values
        attn_output = torch.bmm(attn_weights, V)  # [batch, seq_len, d_internal]
        # Project back to d_model
        attn_projected = self.out_lin(attn_output)  # [batch, seq_len, d_model]

        # Residual connection 1
        x_res1 = x + attn_projected  # [batch, seq_len, d_model]

        # Feed-forward network (applied position-wise)
        ff_hidden = self.activation(self.ff_hidden(x_res1))  # [batch, seq_len, d_internal*2]
        ff_out = self.ff_out(ff_hidden)  # [batch, seq_len, d_model]

        # Residual connection 2
        output = x_res1 + ff_out  # [batch, seq_len, d_model]

        if not batched:
            output = output.squeeze(0)  # [seq_len, d_model]
            attn_weights = attn_weights.squeeze(0)  # [seq_len, seq_len]

        return output, attn_weights


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
def train_classifier(args, train, dev):
    """
    Train a Transformer classifier on the provided LetterCountingExample lists `train` and `dev`.

    Returns the trained model.
    """
    # Hyperparameters (these are conservative values likely to train quickly on this simple task)
    vocab_size = 27  # a-z plus space (driver constructs Indexer accordingly)
    num_positions = 20
    d_model = 64
    d_internal = 32
    num_classes = 3
    num_layers = 1  # single layer is enough usually; you can increase to 2 or 3

    # Instantiate model
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()  # expects log-probs

    num_epochs = 30
    best_dev_acc = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        random.seed(epoch)
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)

        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            # Forward pass
            model.zero_grad()
            log_probs, _ = model.forward(ex.input_tensor)  # log_probs shape: [seq_len, num_classes]
            # NLLLoss expects input shape [N, C] and target shape [N]
            loss = loss_fcn(log_probs, ex.output_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate on dev set after epoch
        model.eval()
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for ex in dev:
                log_probs, _ = model.forward(ex.input_tensor)
                preds = torch.argmax(log_probs, dim=1).cpu().numpy()
                gold = ex.output.astype(int)
                num_correct += sum(preds == gold)
                num_total += len(gold)
        dev_acc = float(num_correct) / num_total
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc

        print("Epoch %d: loss=%.4f dev_acc=%.4f" % (epoch + 1, epoch_loss, dev_acc))
        model.train()

        # Early stopping criterion to meet coursework target
        if dev_acc >= 0.95:
            print("Early stopping: reached dev accuracy >= 95%%")
            break

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
