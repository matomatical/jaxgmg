# jaxgmg cleanup notes

Independent review of the `jaxgmg` research codebase (branch `cleanup2`, off
`experiments`), prepping a refactor + tests ahead of new research. Produced by reading
the RLC 2025 paper (*Mitigating Goal Misgeneralization via Minimax Regret*) and auditing
all ~27.8k lines across the environments/procgen, baselines/UED, and CLI subsystems.

## Read in this order

1. **`01-paper-summary.md`** — the science the code implements (MEV vs MMER, the 3 envs,
   DR/PLR⊥/ACCEL, the two regret estimators, the oracle math, hyperparameters) + a
   paper↔code map. Start here for *why* the code is shaped this way.
2. **`00-codebase-model.md`** — the architecture model: layered diagram, the training
   critical path, core data structures, a per-module health map, key invariants, and an
   overall assessment. The "detailed model of the codebase".
3. **`02-cleanup-plan.md`** — the deliverable: ranked key issues (severity × effort), a
   6-phase plan (net → subtract → consolidate → polish), a concrete test strategy for the
   tricky parts, and open questions for Matthew.

## Matthew's input

- `mfr-wishlist.md` — Matthew's vision for the eventual **rewrite** (first-class reward
  functions; dep migration to jaxtyping/strux/hijax/tyro). Mostly *post-cleanup*; its
  testing wishes are folded into `02-cleanup-plan.md`'s test strategy.

## Evidence (skim as needed)

- `raw-audit-environments.md` — environments/ + procgen/ deep dive (file:line).
- `raw-audit-baselines.md` — baselines/ + autocurricula/ deep dive (the science core).
- `raw-audit-cli.md` — cli/ + util.py + wrappers/ deep dive.

## TL;DR

Algorithmic core is sound; the debt is accreted research drift: ~4.3k lines of verified
dead code (~15%), heavy copy-paste (8-way `cli/train.py`, corner→dish→pile env fork,
plr↔accel buffers), a few **broken-but-reachable** paths that silently disable experiments
(dish proxy solver, minigrid solver), and **zero tests** over subtle correctness-critical
code (Floyd–Warshall oracle, keys Dyck oracle, GAE, regret estimators). Plan: tests first,
then delete + fix, then consolidate — keeping the paper reproducible throughout.
