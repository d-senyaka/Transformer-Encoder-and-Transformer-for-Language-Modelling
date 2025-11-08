# member2_attention_tests.py
# Member 2 – Self-Attention Verification (Evaluation-mode Testing)

import torch
from transformer import TransformerLayer, make_causal_mask

def _attention_sanity_tests():
    """
    Focused tests for verifying the correctness of self-attention in TransformerLayer.
    Runs in eval() mode to disable dropout scaling for deterministic checks.
    """
    B, L, D = 4, 20, 64
    x = torch.randn(B, L, D)

    # ---- Basic attention check ----
    layer = TransformerLayer(d_model=D, d_internal=D, dropout=0.1)
    layer.eval()  # ✅ disable dropout scaling for testing
    y, A = layer(x)

    assert y.shape == (B, L, D), f"Unexpected output shape: {y.shape}"
    assert A.shape == (B, L, L), f"Unexpected attention shape: {A.shape}"

    row_sum = A[0].sum(dim=-1)
    print("Row sums (first 5):", row_sum[:5])
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3), \
        "Attention rows should sum to ~1"
    print("✅ Basic attention sanity passed")

    # ---- Causal mask check ----
    mask = make_causal_mask(L)
    y2, A2 = layer(x, mask=mask)

    assert A2.shape == (B, L, L), f"Unexpected masked attention shape: {A2.shape}"
    # confirm upper-triangular masking works (zeros above diagonal)
    upper = torch.triu(A2[0], diagonal=1)
    assert torch.all(upper == 0) or torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), \
        "Causal mask not applied correctly"
    print("✅ Causal mask applied OK:", A2.shape)


if __name__ == "__main__":
    _attention_sanity_tests()
