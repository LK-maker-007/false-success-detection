# Review: code correctness

Date: 2026-09-13. Read-only, nothing modified.

### 1. Task-level leakage in `stratified_split`; the core claim does not survive removing it
`fsdetect/detect.py:56-60` splits rows at random. 10,268 rows but only 278 distinct
`(domain, task_id)` pairs, ~37 rows per task. Every tau2 task appears in train and test in every
split. Used by ablation, detectors, robustness, portability, adversarial, corpus_audit.
`heldout.py` groups by submission, which does not fix it: all 278 tasks appear in all 12 submissions.

`GroupShuffleSplit(test_size=0.15)` grouped on `(domain, task_id)`, mean of seeds 42-46:

    split                     text     structural   tool names   arguments
    random row (shipped)      0.9441      0.9215       0.8893      0.9056
    group = submission+task   0.9450      0.9211       0.8851      0.9018
    group = task              0.9008      0.8299       0.8312      0.8106

Grouping by submission+task changes nothing, so it is task identity that leaks, not trial
repetition. Task-disjoint: structural loses 9.2 points against text's 4.3, and tool names beat
arguments (0.8312 vs 0.8106) on the mean and on 3 of 5 seeds individually.
**The core claim is an artefact of the split.**

### 2. The abstract's headline numbers belong to a different comparison
Arguments vs tool names on the shipped split, 10 seeds:

    mean +0.0075 +/- 0.0144, arguments win on 7/10 seeds
    DeLong p > 0.05 on 8 of 10 seeds

Not +0.031, not 10/10, not significant, even before fixing finding 1. The Holm family tests five
pairs and omits precisely the one the paper claims.

### 3. `structural` names two different detectors across result files
`adversarial.py:42` fits tool-names-only and writes it into `clean["structural"]` (0.8865).
`robustness.json` reports `structural_clean` 0.9216. Same key, two detectors, 0.035 apart. The
crossover in adversarial.json is declared against the weaker one.

### 4. `attack_a_flip_rate` computed on two populations under one name
robustness filters degenerates first: 218 population, 8.26% baseline. adversarial does not: 1616
population, 1.18% baseline, diluted 7x by 1,398 copies of the transfer token.

### 5. `heldout.py:120` hardcodes a number another script computes
`random_split_reference` is ablation.json's means, copied by hand, printed as the first row of the
section 5.1 table. detectors.json reports different numbers for the same protocol at seed 42.

### 6. `delong_test` has two return shapes; one crashes its caller
`stats.py:48-49` returns 5 keys in the degenerate branch, 7 otherwise. `ablation.py:162,171` read
`se` and `ci95` unconditionally: KeyError. `portability.py:85` splats and silently emits a row with
no interval. Two tests exercise the branch and assert only on `p`, cementing it. `nan <= 0` is
False, so with m==1 or n==1 the guard does not fire and z is nan.

### 7. `api.fit` fits its threshold in-sample and reports the training-set flag rate

    n_train  reported  delivered
        100    0.1000     0.0405
        200    0.1000     0.0737
        400    0.1050     0.0854
        800    0.1000     0.1042
       2000    0.1010     0.0885

At n=100, the case the docstring targets, it advertises 10% and delivers 4%. The 1e-16 effect the
comment does address is 250x smaller. Separately, `api.featurize` has no evaluation anywhere in
results/: it is neither `otel_zero_config` nor `structural`, drops `pay:` and `nids:`, and emits
ok/err per call rather than per tool message. The shipped API's feature set is unmeasured.

### 8. `argument_tokens` is missing the `function.name` fallback its siblings have
`corpus.py:122` uses `call.get("name")` only. `action_tokens:58` and `otel_tokens:155` both fall
back to `function.name`. On any OpenAI-shaped export every call is skipped and the function returns
`"nids:0 nmut:0"` for the entire corpus. AUROC collapses to ~0.5 with no error, no warning, no drop
counter. This is the feature set the core claim depends on.

### 9. `ncalls:` and `nmut:` mean different things in tau2 and AppWorld
`corpus.py:178` ncalls = min(count, 20). `appworld.py:24` ncalls = min(count // 10, 12). So
`ncalls:5` is "5 calls" in tau2 and "50-59 calls" in AppWorld. `nmut` counts argument keys in
corpus.py and mutating calls in appworld.py. Section 5.4's "not a degenerate comparison" defence
rests on 13 of its 18 shared tokens being `ncalls:*`, a collision between incompatible encodings.

### 10. Paper numbers with no producing script
"851 distinct outputs / 90.6% / 0.943 character similarity"; "37,558 features" (detectors.json says
30000, the cap); the ">=100 claims" filter in section 2.3 (applied by hand; the script's own 15
submissions give range 19.5-100%, median 74.4%).

### 11. Section 5.1 states a failure rate contradicting its own source and next paragraph
38.5% vs heldout.json's 155/839 = 18.5%, against a stated range of 3.3-18.8%.

### 12. `precision_at_flag_rate` breaks ties toward the positive class (latent)
`detect.py:103` uses `argsort(-scores)`, ascending index order on ties, and `detect.py:51` orders
rows positives-then-negatives, so index order is label order. All-tied scores report P@10% = 1.0
against a true 0.1. `train_test_split` shuffles so most scripts are accidentally safe;
`GroupShuffleSplit` and `LeaveOneGroupOut` return sorted indices, so heldout.py sits on the biased
path. Checked all 12 LOSO folds: 1 boundary tie each, reported numbers unaffected. One degenerate
feature set from firing.

### 13. The validation split is dead
70/15/15 returned; every caller writes `train, _, test`. 1,540 rows discarded, and the provenance
record implies they were used.

### 14. `corpus_audit.ablate` dedup is order-dependent and biases toward FS
One `seen` set across FS, HF, AMB, TS in that order, so a closing in both FS and TS keeps the FS
copy. 68 repeated closings covering 2,096 rows.

### 15. Dead code and duplicated definitions
`api.py:16` `_ID_LIKE`'s 25+ alternative is unreachable behind a `len <= 24` short-circuit. Three
identifier regexes for one concept. `corpus_audit.py:19` re-declares DEGENERATE_RE verbatim from
`attack.py:58`. `appworld_prevalence.py:42` writes a counter never read. The `function.name` fallback
is dead on this data. `successful_tools` computed for all 12,859 records, read only by mechanism.py.

### 16. Tests that cannot fail or assert the implementation
`test_api.py:68-73` asserts a 4.5x band and a tautology; neither would catch the 2.5x error in
finding 7. `test_stats.py:31-36` asserts AUROC <= 1.0, true by definition. Two tests exercise the
degenerate DeLong branch and assert only on `p`. No test covers `stratified_split`, `fit_combined`
end to end, `endpoint_tokens`, `otel_tokens`, `completion_claim`, or `appworld.load`.

### 17. Minor
Unguarded divisions in mechanism.py:37, appworld_prevalence.py:77, corpus_provenance.py:89. Set
iteration at corpus_provenance.py:118 varies JSON key order between runs. `adversarial.py:75`
`distinct_rewrites` overstates diversity ~380x. `entity_of` returns declaration order, so a message
naming both a flight and a reservation always yields "reservation". `min(int(value), 10)` truncates
floats and has no lower bound. Two places mutate caller dicts in place. AppWorld's explicit-status
corpus runs 10 GroupShuffleSplit seeds over 13 groups, so the seeds draw from few distinct splits.

## Checked and fine
meet_in_the_middle is tractable and "exhaustive" is accurate. Logistic regression converges
everywhere (n_iter 20-29 against max_iter 2000-3000). mcnemar_exact receives booleans at every call
site. corpus_audit's exactly-unchanged 0.8865 is a genuine rounding coincidence across two different
test sets, not a filtering bug. Tool-call id matching is correct for this data. LEXICON_SWAPS is
ordered longest-first and the optional-group back-reference substitutes empty rather than raising.
`labels["HF"]` is disjoint from FS union TS, so attack A's population is genuinely unseen.
`appworld_detectors.py` splitting by submission is the one place the split protocol is sound.
