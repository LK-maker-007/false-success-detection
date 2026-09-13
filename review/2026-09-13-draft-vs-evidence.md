# Review: draft against its own evidence

Date: 2026-09-13. Read-only. All eight cited arXiv IDs verified on the web, plus the OTel registry
and tau2-bench issue #384.

## Tier 1, these sink the paper

### 1. No tau2 result is task-disjoint, and the draft says that protocol is invalid
draft:106 justifies AppWorld's group split because "a random split would place the same task's trace
on both sides". draft:51 then gives tau2 a bare "Stratified 70/15/15 split". `task_id` is loaded at
`corpus.py:308` and never used for grouping.

    claude-3-7-sonnet  sims 200 tasks 50 trials/task {4: 50}
    claude-opus-4-5    sims 200 tasks 50 trials/task {4: 50}
    overlap 50 of 50

Up to 48 trajectories of the same task, same request, same target DB state, scattered across
train/val/test. Section 5.1, sold as the strict protocol, holds out submissions, so every held-out
agent's 278 tasks were seen via the other eleven. There is no task-disjoint number anywhere.

The reference work's headline detector number IS task-disjoint (verified from 2606.09863's abstract:
"task-disjoint AUROC 0.83 on tau2-bench and 0.95 on AppWorld"). Section 7 blames non-comparability
on corpus size alone. The protocol difference is larger and flatters this paper.

### 2. The abstract's two headline numbers are not the comparison the abstract claims
+0.031 is `structural_vs_tool_names` (names+arguments vs names). Arguments alone vs names alone is
+0.0075, inside one SD of either. "10/10 seeds" exists in the JSON only as `combined_beats_text_seeds`
and `both_beats_endpoints_seeds`; there is no per-seed win count for arguments-vs-names on either
benchmark. Section 2's heading "Arguments detect it, names do not" is false on its own table.

### 3. Section 5.1's 38.5% is wrong under every reading
Quoted as the highest failure rate; two paragraphs later the same section says the range is 3.3% to
18.8%. As an FS rate that fold is 155/839 = 18.5% and the highest is 18.8%. As a reward failure rate
it is not the highest either (gpt-4-1-mini 45.9%). "The oldest model" is not unique; four
submissions share the date.

### 4. The evaluation population excludes every hard negative
1,616 HF + 975 AMB = 2,591 trajectories, 20% of the corpus, dropped. That removes exactly the class
a deployed detector must discriminate against, and section 5 is titled "What a deployment would
face". Every P@10% in the paper sits on that population. Section 7 does not list this.

## Tier 2, factual errors and untagged numbers

5. Section 3's 13.3% omits the 8.26% unmodified baseline in the same JSON file, and the paired
   McNemar the draft does not cite (a_only 12, b_only 0, p 0.000488).
6. Section 5.2 reports a sign backwards (+0.0045 where the JSON says -0.0045, native is behind),
   and mislabels which two of three comparisons favour "no list". Then generalises from p 0.232.
7. Numbers with no results file behind them: "37,558 features" (nowhere in results/); "851 distinct
   outputs / 90.6% / 0.943 mean character similarity" (untagged, nowhere, and 90.6% is reused as
   load-bearing under a tag that does not contain it); the airline task 16 and task 9 worked
   examples (no script emits them); ToolScan's seven types (no [C]); "pooled 55.9%" is over 15
   submissions while the sentence scopes to 11 (which give 55.7%).
8. Section 6's one supporting example is contradicted by the paper it cites. The Butterfly Effects
   Australia/US case is `Tool Return: Unsupported 'US' region parameter` -- the call was rejected.
   The distinction the paragraph is built on loses its only evidence.
9. Section 4.2 reports the milder of two ablations and mislabels the operation. The third row,
   `deduplicated`, costs text 0.0121 and does move the tool-names detector (0.8865 to 0.8817),
   which the sentence asserts is 0.0000. "Removing harness control tokens" is actually deleting
   1,979 whole trajectories, all negative.
10. Section 3's "three generic sentences" is contradicted by `corpus_audit.json`: 43 distinct
    strings per style over 1,616, entity unresolved in 81.7%.
11. Section 1 welds two mutually inconsistent fields: `individual_checks_failing` says 656 action
    failures, `reward_components_failing` says 6 have ACTION. DB reconciles to within 13; ACTION
    does not reconcile at all.
12. Missing prior art. **arXiv 2605.10448** "Can Agent Benchmarks Support Their Scores?" (May 2026)
    motivating example: a task asks whether Alice's address changed, the check only verifies the
    agent clicked Save, the agent may have modified the wrong record. That is section 1's mechanism
    and section 4's contribution, four months earlier, and it is already in the project memory note.
    Also **2606.08162**, which CLAUDE.md says any adversarial framing must answer in the intro.
13. Section 6 overstates the gap to 2511.04032. Its feature list includes Prompt and Context,
    "intermediate inputs/outputs", not just "sequence and volume".
14. Causal and significance language beyond what was measured: "The cause is not instability"
    before a bare correlation over 12 dependent points; "Both are chance" with no interval;
    "explains three otherwise puzzling observations" which are never enumerated; the abstract
    states the mechanism as measured on "two agent benchmarks" when mechanism.json is tau2 only.
15. Limitations too weak. OTel content capture is off by default, not an edge case. No trivial
    baseline anywhere: FS mean 91.3 words vs TS 70.2, and nobody reports what word-count-only
    scores. Section 3 compares an LR text detector to the reference's XGBoost throughout without
    noting the model differs.

## Clean
Section 4.1 (every number checks, reference composition verbatim correct), section 4.2 (issue #384
verified verbatim including file:line and all eight percentages), section 5.3, section 5.4,
section 2.1's DeLong table and variance-source paragraph, section 2.3's prevalence range.

Almost no duplication after the reorganisation; three residual cross-references only.
