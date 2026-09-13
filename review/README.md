# Reviews

Five independent read-only reviews of this repo, all run 2026-09-13 against commit `fb2ebb6`, each
on a separate surface.

| file | scope |
|---|---|
| `2026-09-13-code-correctness.md` | leakage, silent wrong answers, dead code, tests that cannot fail |
| `2026-09-13-statistics.md` | DeLong and Holm implementations, variance conflation, multiple testing |
| `2026-09-13-reproducibility.md` | clean-checkout rebuild, CI coverage, corpus pinning, secrets |
| `2026-09-13-draft-vs-evidence.md` | every number traced, every tag checked, every citation fetched |
| `2026-09-13-purpose-and-route.md` | is the problem real, is anyone already solving it, is this a dead end |

## The finding that matters

Three of the five independently identified task-level leakage in `stratified_split`. The one that
measured it found the core claim reverses: under a task-disjoint split, tool names beat arguments,
0.8312 to 0.8106. Every tau2 AUROC in the paper is inflated, and the abstract's headline numbers
describe a comparison the repo never ran.

The reviewer's harness reproduced the repo's published figures exactly before diverging, which is
what makes the divergence attributable to the protocol rather than to their setup.

## What survives

The evaluator-validity findings, verified clean by two reviewers independently:

1. tau2-bench's reward auto-passes a gate on 88% of airline simulations, traced to two evaluator
   lines, with maintainer issue #384 open and unanswered for ten weeks
2. the reference paper's corpus cannot be reconstructed from the source it names
3. random splits on this benchmark leak tasks and inflate AUROC by up to 9 points
4. the published "dual control suppresses false success tenfold" result is confounded with telecom
   being the only domain whose reward function is not broken
