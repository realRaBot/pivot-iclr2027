# Exact contribution and provenance map

The FAST-VOLLIB package is included as source in this artifact. The tensor-native
Jäckel backend was developed in the FAST-VOLLIB framework and is included here
so the implementation can be audited directly.

| Component | Prior mathematics / infrastructure | Paper-associated implementation | Artifact path |
|---|:---:|:---:|---|
| Jäckel/LBR scalar algorithm | Yes | No | cited and reimplemented in `jackel_iv.py` |
| Vectorized normalized-Black/LBR core | No | Yes | `src/fast_vollib/jackel/jackel_iv.py` |
| Heterogeneous NumPy/PyTorch/JAX backends | No | Yes | `src/fast_vollib/jackel/*_backend.py` |
| Fused Triton LBR forward kernel | No | Yes | `src/fast_vollib/jackel/triton_kernels.py` |
| Inverse-vega/IFT identity | Yes | No | implemented, not claimed as a new theorem |
| PyTorch autograd interface and implicit VJP | No | Yes | `src/fast_vollib/jackel/differentiable.py` |
| JAX `custom_vjp` interface | No | Yes | `src/fast_vollib/jackel/differentiable_jax.py` |
| Invalid-domain and upstream-aware low-vega contract | No | Yes | both differentiable modules + tests |
| Smooth gate and sentinel-protected objectives | No | Yes | `experiments/.../aux_losses.py` |
| Matched forward+backward benchmark | No | Yes | `bench_differentiable_iv.py` |
| Downstream integrations and equivalence probe | No | Yes | experiment scripts |

The remainder of `src/fast_vollib/` is included as supporting framework code
needed to execute the exact snapshot. It is not presented as a contribution of
this paper.
