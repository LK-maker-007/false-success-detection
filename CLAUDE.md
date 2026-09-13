# Research: false-success detection in LLM agents

## Current goal

Adversarial robustness of false-success detection, on our own pinned corpus.

The original goal was to reproduce AUROC 0.913 from arXiv 2606.09863. That criterion fired on 2026-09-13 and was resolved by proof of impossibility, not by a match: his corpus (9,876 = 8,146 success / 1,730 failure, 17.5% failure) cannot be constructed from the leaderboard trajectories at any granularity. Exhaustive meet-in-the-middle over all 37 files returns 0 solutions; the pooled failure rate is 26.8%. Files are unmodified since 2026-03-23, so this is not drift. See the memory note for the full evidence chain.

All figures mean +/- std over seeds 42-51, `scripts/robustness.py`, sklearn 1.5.2, stratified 70/15/15:

- text detector, FS vs TS, closing message only: **0.9458 +/- 0.0074**
- structural detector, tool names + call arguments, no message text: **0.9230 +/- 0.0081**
  (tool names alone were 0.8924; arguments are worth +0.031)
- text under `swap_lexicon`, the content-preserving attack: **0.9212 +/- 0.0070**
- text under the template styles: 0.67 to 0.84, but templates replace the message rather than evade the detector
- 12,859 records, 851 FS / 9,417 TS, excluding 1,616 HF and 975 ambiguous

Corpus pin, which the paper does not have and this does:
- S3 `sierra-tau-bench-public`, keys in `data/tau2/_bucket_keys.json`, all files `Last-Modified 2026-03-23`
- leaderboard membership = `manifest.json` at tau2-bench commit `c7e436f33`

**Kill criterion fired, then reopened to parity, 2026-09-13.** With tool names alone the attacked text
detector never fell below structural, 0 of 10 seeds. Adding call arguments lifted structural to 0.9230
against attacked text at 0.9212, and the crossover now lands on 5 of 10 seeds. Five of ten is a coin flip,
so the honest claim is parity, not superiority: invariance is roughly free, it does not win. The template attacks cross on 10 of 10 seeds, but they cross because replacing a 91-word message with one of
three generic sentences is a distribution shift, not an evasion. Do not quote them as an attack result.

What survived the kill:

1. The text detector is not a lexical rule. `swap_lexicon` removes the FS_ASSERT_RE trigger from 90.6% of
   FS closings while preserving 94.3% of characters, and costs only 2.5 AUROC points. Appendix H's 20% flip
   rate invites the opposite conclusion.
2. It still misses 86.7% of content-preserving false successes. `assert_completion` excises the admission
   from 218 substantive HF closings, keeps the body, and flips the detector on only 13.3% +/- 1.5%.

Open question that follows directly: does the structural detector flag those 218 rewritten failures? It
cannot see the rewrite, so its score is unchanged, and they are all reward==0. If it flags them where the
text detector does not, the two are complementary rather than substitutes.

## Anchor paper

arXiv 2606.09863, "From Confident Closing to Silent Failure: Characterizing False Success in LLM Agents". ICML 2026 FAGEN workshop. Single author, Laksh Advani, University of Colorado.

Read the appendices before proposing anything in this space. Two prior candidate directions died to results already in §4.2 and Appendix H/I.

- Appendix A: the two regexes, `FS_ASSERT_RE` and `HONEST_FAILURE_RE`. FS if the first matches and the second does not; HF if inverse; ambiguous if both or neither.
- Appendix C: ngram (1,2), max_features 30000, min_df 2, sublinear_tf, C=1.0, class_weight balanced.
- Appendix H: adversarial rewrite flips XGBoost 10/50 (20%), judges 11/45 (24%). Rewrites preserve tool calls and rewards.
- Appendix I: XGBoost degrades with calibration (0.657 to 0.538 at K=100), LR improves to 0.827. The authors explain it as tree overfitting on the small target set.

Precision is 50% at a 10% flag rate. The authors state neither detection nor LLM-judging has sufficient precision for autonomous deployment.

## Data

- tau2-bench: Sierra's public leaderboard, 8 model families, 9,876 trajectories after cleaning. Each trajectory carries the user-simulator request, the full tool-grounded response sequence, and a programmatic reward.
- AppWorld: publicly released experiment outputs, 4 model families. Abstract says 1,879 trajectories, appendix says 8,190. Resolve before relying on either.
- The pipeline is reproducible from public sources without author cooperation.

## Known constraints

- tau2 and AppWorld detectors cannot transfer to each other. Zero text feature overlap: tau2 features are natural-language bigrams, AppWorld's are API method-endpoint tokens.
- Cross-domain transfer is already reported in §4.2. It is not an open question.
- Structural trace features for silent-failure detection have prior art in arXiv 2511.04032. The unclaimed contribution is adversarial invariance, not the features themselves.
- arXiv 2606.08162 argues silent failure is intrinsic and non-adversarial. Any adversarial framing must answer it in the introduction, not the rebuttal.
