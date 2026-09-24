# PIVOT: differentiable Jäckel implied-volatility operator

Anonymous code artifact for ICLR 2027 submission 8150 (PIVOT).

Canonical code repository: https://github.com/realRaBot/pivot-iclr2027

```bash
git clone https://github.com/realRaBot/pivot-iclr2027.git
cd pivot-iclr2027
```

PIVOT combines a tensor-native implementation of Jäckel's
*Let's Be Rational* (LBR) forward solve with a custom implicit backward,
explicit invalid-domain behavior, and low-vega training semantics. This
archive includes the source that was developed in the FAST-VOLLIB framework
instead of presenting FAST-VOLLIB as an opaque external dependency.

## What is prior and what is contributed

Jäckel's LBR mathematics and the identity
\(\partial \sigma_{\mathrm{imp}}/\partial P=1/\mathrm{Vega}\) are prior
knowledge. The paper-associated implementation work is:

1. a heterogeneous batched LBR backend for NumPy, PyTorch, JAX, and Triton;
2. PyTorch `autograd.Function` and JAX `custom_vjp` layers that keep the
   branch-heavy solve out of the backward graph;
3. invalid-input, low-vega, sentinel, and smooth-gate semantics;
4. matched systems/downstream benchmarks and reproduction harnesses.

See [CONTRIBUTION_MAP.md](CONTRIBUTION_MAP.md) for the exact path-level split.

## Key paths

```text
src/fast_vollib/jackel/jackel_iv.py           vectorized LBR core
src/fast_vollib/jackel/{numpy,torch,jax}_backend.py
src/fast_vollib/jackel/triton_kernels.py      fused GPU forward
src/fast_vollib/jackel/differentiable.py      PyTorch implicit VJP
src/fast_vollib/jackel/differentiable_jax.py  JAX custom_vjp
tests/test_jackel/                             parity + gradient contracts
experiments/.../bench_differentiable_iv.py    matched systems benchmark
experiments/.../train.py                      IV-output controlled study
experiments/.../train_price_output.py         genuine price-output use case
results/                                      compact aggregate manifests
```

## Reproduce without licensed data

```bash
uv sync --locked --extra jax --python 3.12
uv run pytest -q

# CPU-friendly operator smoke:
uv run python examples/synthetic_benchmark.py \
  --device cpu --dtype float64 --batches 1000 10000 --repeats 3

# Matched PIVOT/unrolled/generic-IFT/same-forward control:
uv run python experiments/hyperiv_price_iv_aux/scripts/bench_differentiable_iv.py \
  --smoke --device cpu --dtypes float64
```

The full GH200 command replaces `--smoke --device cpu` with
`--device cuda --dtypes float32 float64 --batches 1000 100000 1000000
10000000 --repeats 7`.

## Downstream synthetic smoke

Generate a licensed-data-free fixture and run one epoch:

```bash
uv run python \
  experiments/hyperiv_price_iv_aux/scripts/make_synthetic_fixture.py

uv run python experiments/hyperiv_price_iv_aux/scripts/train.py \
  --config experiments/configs/hyperiv_price_iv_aux.toml \
  --variant price_rt_aux --seed 1 --epochs 1 --data-tag synthetic \
  --tag smoke_rt --no-wandb
```

For the price-output experiment:

```bash
uv run python experiments/hyperiv_price_iv_aux/scripts/train_price_output.py \
  --config experiments/configs/hyperiv_price_iv_aux.toml \
  --variant pivot_iv --lambda-iv 0.1 --tau 1e-6 \
  --seed 1 --epochs 1 --data-tag synthetic --tag smoke_price \
  --wandb-mode disabled
```

These smoke runs check execution only; they do not reproduce the reported
licensed-data results. See [experiments/README.md](experiments/README.md)
for the controls, seed protocol, and limitations of the supplied summaries.

## Licensed data

OptionMetrics/WRDS rows are not redistributed. The public artifact contains a
synthetic fixture generator, complete column schema, seed lists, scripts,
configuration, and aggregate outputs. Users with licensed data can place
`<tag>_train.parquet` and `<tag>_test.parquet` under
`experiments/hyperiv_price_iv_aux/prepared/`.

## Anonymous review and attribution

This repository is the code artifact; the paper is submitted separately.
The [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
permit an anonymous repository link or an anonymized supplementary code ZIP.
No author-specific paper citation or profile link is included during review.
The existing anonymous MIT license is retained, as are third-party notices
required by the included source.

Prior algorithm: Peter Jäckel, *Let's Be Rational*, Wilmott 2015(75),
40–53, [doi:10.1002/wilm.10395](https://doi.org/10.1002/wilm.10395).
This attribution identifies prior work, not the submission authors.

See [VALIDATION.md](VALIDATION.md) for local checks and numerical limitations.
