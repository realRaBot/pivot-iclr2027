# Local validation

Checked on 2026-09-24 with Python 3.12.14 on macOS ARM64 (CPU), using
`uv.lock`: PyTorch 2.14.0, JAX/JAXlib 0.11.2, NumPy 2.5.3, and SciPy 1.18.1.

| Check | Result |
|---|---|
| Full pytest suite, including PyTorch, JAX and LBR reference tests | 109 passed |
| README synthetic operator benchmark, float64, batches 1,000 and 10,000 | Completed |
| Matched systems benchmark, CPU float64 smoke | All four methods and finite differences completed |
| Synthetic fixture generation | 1,152 training rows and 288 test rows |
| IV-output roundtrip auxiliary, seed 1, one epoch | Completed; checkpoint epoch 1; zero NaN-gradient steps |
| Price-output PIVOT auxiliary, seed 1, one epoch | Completed; zero NaN-gradient steps |
| Wheel build | Passed; both `pivot` and `fast_vollib` included |
| Python/TOML syntax and scheduler shell syntax | Passed |

The one-epoch IV-output check exercises a correction to checkpoint selection:
short runs now save trained weights instead of restoring initialization. Runs
of six or more epochs retain the supplied five-epoch selection warmup.

## Interpretation and limits

The operator smoke reported maximum IV roundtrip absolute error 2.155e-2 and
mean 1.062e-5 across finite outputs, including ill-conditioned inputs; the
maximum relative gradient discrepancy on its well-conditioned subset was
3.983e-3. These are diagnostics, not a blanket machine-precision guarantee.
The matched benchmark's finite-difference check reported maximum relative
discrepancy 6.236e-4 over 1,679 rows. Its zero forward error for PIVOT is
relative to the same float64 solver, not an independent oracle.

One-epoch training checks validate execution and checkpoint handling only.
They do not establish convergence, the benefit of any objective, or replication
of the historical summaries in `results/`. No CUDA/Triton/GH200 checks or
licensed-data experiments were run in this local validation. No linter is
configured in the supplied project.
