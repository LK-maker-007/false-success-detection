# Review: statistical validity

Date: 2026-09-13. Read-only, nothing modified.

**Verdict:** DeLong and Holm are correct. Almost everything built on top of them is either testing a
different quantity than the paper claims, or reusing the split that chose the features.

### 1. The headline claim is never tested, and when tested it fails
The paper claims "tool-call argument features predict task-level failure better than tool-name or
endpoint features". That comparison appears in no test in the repo. `ablation.py:114-120` tests
`structural_vs_tool_names` and `structural_vs_arguments`; `structural` = names PLUS arguments.

The abstract's +0.031 is `structural - tool_names`. The claim's own quantity,
`arguments - tool_names`, is +0.0075, four times smaller. The missing DeLong, run:

    seed 42: args 0.9094  names 0.8865  diff +0.0229  p 0.0446
    seed 43: args 0.9029  names 0.8856  diff +0.0173  p 0.0669
    seed 44: args 0.8897  names 0.8829  diff +0.0067  p 0.5260

Uncorrected p=0.045 on the confirmatory seed; does not survive the paper's own Holm family of five.
AppWorld is the same substitution: +0.036 is `both - endpoints`; `arguments - endpoints` is +0.0271.

### 2. The text baseline is partly the labelling function
Positives are defined as `FS_ASSERT_RE and not HONEST_FAILURE_RE` on the closing message. Negatives
are not filtered at all. The text detector is then trained on that same closing message.

    positives matching FS_ASSERT:        1.0000 (by construction)
    negatives matching FS_ASSERT:        0.3708
    positives matching HONEST_FAILURE:   0.0000 (by construction)
    negatives matching HONEST_FAILURE:   0.2364
    labelling regex used as a classifier:  AUROC 0.8259
    trained text detector:                 AUROC 0.9438

Section 3 claims to test this but `swap_lexicon` is applied only to positives, so it removes the
FS_ASSERT half of the label definition and leaves the HONEST_FAILURE half untouched. The structural
and argument detectors read no text and are clean.

### 3. The confirmatory split was used for feature selection
The val split is discarded at eight of nine call sites and never scored. The grounding feature was
dropped on the basis of seed-42 test AUROC (0.9216 = `detectors.json structural.auroc`). Seed 42 is
then designated confirmatory for all five Holm-corrected DeLong tests and all three portability
tests. `detect.py:22` sources the winning configuration to `results/feature_ablation.json`, which
does not exist.

### 4. The AppWorld group split does not do what section 2.2 says
Grouping by submission does not prevent a task appearing on both sides; it guarantees it. It does
not fully control agent identity either: the group key is the experiment directory, encoding
agent x split, so 15 submissions collapse to 9 distinct agents.

### 5. r = 0.932 is 0.893 arithmetic
Precision@10% is hard-capped at base_rate/0.10.

    r(fold FS rate, P@10)      = 0.9318   <- the reported number
    r(fold FS rate, CEILING)   = 0.8933   <- pure arithmetic, no detector
    r(fold FS rate, P/ceiling) = -0.1248
    P/ceiling: mean 0.769 std 0.073

The non-trivial result, detector efficiency flat at 0.77 across a 5.6-fold base-rate range, is
unreported. The 12 folds share 11/12 of their training data and are not independent.

### 6. Two variance sources conflated in sections 2.2 and 5.2
Section 2.1 separates them correctly. Section 2.2's table columns subtract to 0.0338 while the gap
is quoted as +0.0363 from a seed-42 bootstrap. Section 5.2 quotes +0.0045 where the JSON says
-0.0045, sign inverted, and neither number in the test appears in the table above it.

### 7. "Statistically indistinguishable" from p = 0.232
Accepting the null from a non-significant test. No equivalence margin, no TOST. The CI admits a
1.17-point advantage to zero-config.

### 8. Two disjoint Holm families over the same split
13 formal tests, 8 corrected, in two families that share the seed-42 split and the `structural`
detector. Escaping correction: both `combined_minus_text` bootstraps (duplicate tests of hypotheses
already in the Holm family, with the uncorrected one quoted in the robustness section), both
`both_minus_endpoints`, `crossover_margin`, three `bootstrap_auroc` CIs, one McNemar. Roughly 35
further directional conclusions drawn in prose with no test. At alpha=0.05 over ~50 looks, expect
~2.5 spurious.

### 9. The attack is 12 strings, reported as 1153
`distinct_rewrites: 1153` counts distinct strings across the whole 1541-row attacked column,
including 1413 untouched negatives. Actual distinct rewrites per style: 12, most common covering 29
of 128. `worst_text_auroc` is a min over five transforms with no selection adjustment, and the
selected transform is one section 3 explicitly disowns as invalid.

### 10. 20.4% duplicate closings, no headline number recomputed without them
2096 of 10268 rows share a closing. The largest single group is 1045 identical copies of the
transfer token, all TS, zero FS. Dedup costs text 0.0121. Main tables never recomputed on the
cleaned corpus, and the two ablations are on different test sets so the comparison is unpaired.

### 11. Precision@10% reported without its ceiling on tau2, without its base rate on AppWorld
tau2 ceiling is 0.8312, so 0.6156 is 74% of achievable, not mediocre. On AppWorld positives are the
majority class at 55.1%, so a coin flip gets 0.551. Quoting 0.9363 to 0.9595 beside tau2's 0.5708
to 0.6156 invites a false comparison.

### 12. The AppWorld corpus fork, and a wrong mechanism for the reversal
On the strict corpus the ordering reverses: arguments 0.8325 > both 0.8093 > endpoints 0.7892. The
paper attributes this to sample size, tagged [H]. Sample size does not reverse a sign; the strict
corpus is a different population. The reversal is the direct analogue of tau2's non-significant
`structural_vs_arguments`.

### 13. Two result files disagree about the same experiment
`adversarial.json` attack_a_population 1616, unmodified flip 0.0118. `robustness.json` 218 and
0.0826. Both listed as sources for section 3; neither mentions the other's population.

### 14. `assert_completion` quoted without its own control
13.3% flip against an 8.26% unmodified baseline on the same 218. Marginal effect 5.0 points, not
13.3. McNemar in the same file: a_only 12, b_only 0, p 0.000488. Overstated 2.6x.

### 15. LOSO folds labelled "seeds" and averaged unweighted
`spread()` hardcodes "seeds". Twelve dependent folds of n=242 to n=1050 weighted equally, no CIs.

### 16. Smaller
- `heldout.py:120` hardcodes another script's output as the comparison row.
- Section 2.3 pools over 15 submissions while scoping the sentence to 11. Part of the "four-fold
  spread across agents" is a spread across difficulty splits of the same agent.
- Section 5.4 "both are chance" stated with no interval.
- Section 1's opening finding carries no CI on either rate or the ratio.
- `stats.py:48-49` returns a dict with no `se` and no `ci95` in the degenerate branch.

## Passed
`delong_test` matches sklearn to 1e-16 and an independent fastDeLong transcription to 1.5e-3;
`_midrank` exact against scipy. `holm_bonferroni` correct step-down, reproduces every p_holm.
`mcnemar_exact` standard. `_resample_indices` is the right conditional bootstrap.
