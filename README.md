# Flow-TRAC

Minimal offline RL implementation of behavior-prior-tilted policy improvement with
conditional flow matching.

The implementation keeps two independent flows:

- a behavior flow `mu(a | s)`, pretrained on the Minari dataset and then frozen;
- a Q-tilted flow actor, trained with either resampled or weighted flow matching
  from frozen behavior-flow candidates weighted by `exp(Q(s, a) / lambda)`.

Each flow also learns a deterministic conditional-mean readout. Stochastic ODE
samples still define the TRAC base measure; the mean readout is used only when a
deterministic closed-loop action is requested.

Twin target critics use the same frozen behavior flow as the base measure in the
TRAC Bellman backup. No flow likelihood or CNF divergence is required.

## Install

```bash
conda activate cleanrl
pip install -e .
```

## Train

The defaults target Minari Kitchen Complete and use an 8-step Euler solver:

```bash
flow-trac
```

For a quick wiring check with an already downloaded dataset:

```bash
flow-trac \
  --behavior-pretrain-steps 2 \
  --total-updates 2 \
  --batch-size 8 \
  --num-value-samples 2 \
  --actor-num-candidates 2 \
  --flow-steps 2 \
  --eval-freq 0 \
  --no-save-model
```

Important options:

- `--critic-warmup-steps`: critic-only updates after behavior pretraining.
- `--lambda`: TRAC temperature used by both the Bellman target and actor weights.
- `--actor-mode`: `weighted` (default, flow matching over every weighted behavior
  candidate) or `resampled` (one sampled target per state).
- `--cql-alpha`: conservative critic regularization strength; set to zero to disable.
- `--no-cql-include-uniform`: use only frozen behavior-flow candidates in CQL.
- `--no-cql-include-data-action`: disable the default exact-data-action CQL anchor.
- `--no-actor-updates`: train only the frozen behavior flow and critics for diagnostics.
- `--eval-policy`: `actor`, `prior`, or diagnostic `prior-resample` evaluation.
- `flow-trac-eval --flow-steps N`: override ODE steps when comparing a saved checkpoint.
- `--observation-key`: select one key from Dict observations; use `None` to flatten all keys.

TensorBoard logs are written below `runs/`. In particular, actor diagnostics include
ESS, maximum importance weight, weighted/prior Q, candidate diversity, flow loss,
and path velocity norm.

Weighted flow matching is the default. The original resampled objective remains
available explicitly:

```bash
flow-trac --actor-mode resampled
```

## Reproduced Gaussian TRAC baseline

The original `offline_rl` Gaussian/GMM TRAC implementation is included as a
self-contained `trac/` package. Its Kitchen defaults and update rules are kept
separate from Flow-TRAC so the two implementations can be compared directly.
The defaults reproduce the reference Kitchen run, including CQL `alpha=0.1`;
Flow-TRAC keeps its own CQL default unchanged.

Both implementations use a flat package layout: `agent.py`, policy/model
modules, data helpers, and `train.py`. Legacy nested TRAC checkpoints remain
loadable after the layout change.

```bash
trac-kitchen --cuda
```
