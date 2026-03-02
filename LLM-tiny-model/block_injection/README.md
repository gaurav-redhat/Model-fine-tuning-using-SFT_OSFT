# Block Injection — Code Examples

Self-contained PyTorch demos for each injection strategy described in [`BLOCK_INJECTION_GUIDE.md`](../BLOCK_INJECTION_GUIDE.md).

## File Map

| File | Strategy | Accuracy | Speed | New Params |
|---|---|---|---|---|
| `s1_adapter_modules.py` | Bottleneck adapters between frozen layers | +2–8% | -5% | 0.5–3% |
| `s2_sparse_upcycling.py` | Dense FFN → Mixture-of-Experts | +3–10% | Same | Router only |
| `s3_mamba_block_replacement.py` | Replace attention with SSM (Mamba) | ~Same | 2–3x | Replacement |
| `s4_early_exit.py` | Exit heads at intermediate layers | ~0% | 1.5–3x | <0.1% |
| `s5_speculative_decoding.py` | Draft head for parallel verification | **0%** | 2–3x | 5–10% |
| `s6_dynamic_layer_skip.py` | Per-input layer skip predictors | -0.5–2% | 1.5–2.5x | <0.01% |
| `s7_attention_replacement.py` | Sliding window / linear attention | -1–3% | 1.5–2x | Replacement |
| `s8_cross_attention_injection.py` | Cross-attention for vision input | +New capability | -10% | 10–20% |
| `s9_side_network.py` | Parallel side network with corrections | +2–5% | -10% | 5–15% |
| `s10_prefix_tuning.py` | Learnable soft tokens at every layer | +1–5% | ~0% | 0.01–0.1% |

## Quick Start

```bash
pip install -r requirements.txt

# Run any strategy demo
python s1_adapter_modules.py
python s2_sparse_upcycling.py
python s3_mamba_block_replacement.py
python s4_early_exit.py
python s5_speculative_decoding.py
python s6_dynamic_layer_skip.py
python s7_attention_replacement.py
python s8_cross_attention_injection.py
python s9_side_network.py
python s10_prefix_tuning.py
```

Strategies s2–s10 use toy models and run on CPU. Strategy s1 optionally loads a real HuggingFace model (needs GPU + model download).

## Key Principle

All strategies follow the same pattern:

```
Pre-trained model (FROZEN) + Injected block (TRAINABLE) → Better model
```

No training from scratch required.
