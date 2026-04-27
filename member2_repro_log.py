import json, torch, random, numpy as np, os
from updated_member1_data_pipeline import build_vocab_indexer, build_dataloader
from transformer import Transformer
from torch import nn, optim
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("artifacts", exist_ok=True); os.makedirs("plots", exist_ok=True)

# 1) Data
v, idx = build_vocab_indexer()
train = build_dataloader("data/lettercounting-train.txt", idx, batch_size=1, shuffle=False)
dev   = build_dataloader("data/lettercounting-dev.txt", idx, batch_size=1, shuffle=False)

# 2) Model
model = Transformer(vocab_size=27, num_positions=20, d_model=64, d_internal=64, num_classes=3, num_layers=1)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_f = nn.NLLLoss()

# 3) Quick 3-epoch train (just to produce a reproducible artifact)
for ep in range(3):
    model.train(); tot=0.0
    for xb, yb in train:
        logp, _ = model(xb[0])
        loss = loss_f(logp, yb[0])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"[repro ep {ep+1}/3] loss={tot:.3f}")

# 4) Dev accuracy
model.eval()
correct=0; total=0
for xb, yb in dev:
    logp, _ = model(xb[0])
    pred = logp.argmax(dim=1)
    correct += (pred==yb[0]).sum().item()
    total   += yb.numel()
acc = correct/total

# 5) Save everything needed for the report
torch.save(model.state_dict(), "artifacts/model_member2_demo.pt")
with open("artifacts/run_meta.json","w") as f:
    json.dump({
        "seed": SEED, "d_model": 64, "d_internal": 64, "layers": 1,
        "lr": 1e-3, "epochs": 3, "dev_acc": acc
    }, f, indent=2)

print("Saved → artifacts/model_member2_demo.pt; dev_acc=", round(acc,4))
