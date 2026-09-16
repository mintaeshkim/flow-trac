# Flow-TRAC

Minimal offline RL implementation of behavior-prior-tilted policy improvement with
conditional flow matching.

The implementation keeps two independent flows:

- a behavior flow `mu(a | s)`, pretrained on the Minari dataset and then frozen;
- a Q-tilted flow actor, trained with either resampled or weighted flow matching
  from frozen behavior-flow candidates weighted by `exp(Q(s, a) / lambda)`.

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
- `--actor-mode`: `resampled` (default, one target per state) or `weighted`
  (flow matching over every weighted behavior candidate).
- `--cql-alpha`: conservative critic regularization strength; set to zero to disable.
- `--eval-policy`: `actor`, `prior`, or diagnostic `prior-resample` evaluation.
- `--observation-key`: select one key from Dict observations; use `None` to flatten all keys.

TensorBoard logs are written below `runs/`. In particular, actor diagnostics include
ESS, maximum importance weight, weighted/prior Q, candidate diversity, flow loss,
and path velocity norm.

To train the actor with weighted flow matching while keeping the same frozen prior
and TRAC critic target:

```bash
flow-trac --actor-mode weighted
```
