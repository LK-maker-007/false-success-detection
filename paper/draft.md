# What false success in LLM agents actually is, and what detects it

> **Status: Project Halted on Classifier Ranking Hypothesis.**
> The classifier ranking claim in Sections 2-3 (that tool-call arguments outperform tool names) was refuted under strict task-disjoint splits: the difference collapses to -0.0045 (AUROC 0.8247 vs 0.8292, p = 0.868, 5/10 seed wins; see `results/protocol.json`).
> 
> The empirical audits in Section 4 (the tau2-bench reward auto-pass bug across 88% of airline simulations and the reference corpus reconstruction impossibility) and the deployment constraints in Section 5 remain verified findings.

Draft. Every number carries [V] verified this session with the script that produced it, [C] cited
to a published source, or [H] hypothesis. Untagged numbers are defects.

## Abstract

An LLM agent fails silently when it closes by asserting a task is complete and the environment
state disagrees. On two agent benchmarks we find that this failure is almost never an agent
claiming an action it never attempted. It is an agent performing the right operation with the
wrong arguments, or completing some of the required steps. We show that features taken from
tool-call arguments detect it better than features taken from tool names or API endpoints, on
tau2-bench (+0.031 AUROC, 10/10 seeds) and on AppWorld (+0.036 AUROC, 10/10 seeds, held out by
submission). We also report two defects in how the field measures this: the corpus behind the
reference result cannot be reconstructed from the source it names, and tau2-bench's reward
auto-passes a gate on 88% of airline simulations.

## 1. What false success is

Prior work defines false success as a trajectory whose final message asserts a completed action
that the tool-call history and reward indicate did not occur [C, arXiv 2606.09863 §1]. That
phrasing invites a natural detector: check whether the claimed action appears in the trace. It
does not work, and the reason is the paper's most useful finding.

Across 851 false-success trajectories in our tau2-bench corpus, the operation the closing message
claims is almost always present and successful in the trace. Among trajectories that claim an
operation, the rate of having no matching successful call is 6.0% for false successes and 14.6%
for true successes, a ratio of 0.41. The signal is not weak. It is inverted.
[V, results/mechanism.json]

The reward record says why. Of those 851 trajectories, 93.1% fail the database check and 77.1%
fail at least one action check, while the communicate gate passes in most cases. 85.1% fail on the
database check alone. [V, results/mechanism.json]

Reading examples makes the mechanism concrete. In airline task 16 the user states they are happy
with the original payment method for a refund; the agent refunds to a gift card, the
`update_reservation_flights` call succeeds, and the database check fails. In airline task 9 the
user asks for two reservations cancelled and a third changed to a nonstop flight; the agent
cancels the two, announces success, and both `cancel_reservation` calls succeed.

So the agent calls the right tool. The arguments or the completeness are wrong. This explains
three otherwise puzzling observations at once, and it predicts where a detector should look.

## 2. Arguments detect it, names do not

### 2.1 tau2-bench

Corpus: 12,859 trajectories from 12 leaderboard submissions across airline, retail and telecom.
Positives are the 851 failures whose closing asserts completion, negatives the 9,417 true
successes. Honest-failure and ambiguous trajectories are excluded, following the reference
protocol. Stratified 70/15/15 split, TF-IDF and logistic regression, seeds 42 to 51.
[V, results/detectors.json, results/ablation.json]

| features | AUROC |
|---|---|
| tool names only | 0.8924 +/- 0.0095 |
| call arguments only | 0.8999 +/- 0.0118 |
| both (structural) | 0.9230 +/- 0.0081 |
| closing message (text) | 0.9458 +/- 0.0074 |
| text + structural | **0.9557 +/- 0.0064** |

[V, results/ablation.json]

Significance is tested with DeLong on one designated confirmatory split, seed 42, with Holm
controlling the family-wise error rate across five comparisons. Other seeds are robustness, not
independent confirmations, since they re-split one fixed dataset.

| comparison | diff | DeLong p | p after Holm | significant |
|---|---|---|---|---|
| combined vs text, clean | +0.0107 | 6.83e-03 | 1.37e-02 | yes |
| combined vs text, attacked | +0.0194 | 1.87e-05 | 5.60e-05 | yes |
| structural vs tool names | +0.0351 | 3.84e-08 | 1.53e-07 | yes |
| structural vs arguments | +0.0122 | 1.18e-01 | 1.18e-01 | **no** |
| text clean vs attacked | +0.0186 | 9.75e-19 | 4.88e-18 | yes |

[V, results/ablation.json delong_holm_corrected]

Adding arguments to tool names is significant. **Adding tool names on top of arguments is not**, at
p 0.118 before correction. Across seeds that difference is +0.0231 +/- 0.0121, so the effect may be
real and this split cannot show it. We report the combination because it is what we ran, and we do
not claim it beats arguments alone.

Precision at a 10% flag rate rises from 0.5708 to 0.6156. [V, results/ablation.json]

Two variance sources are kept apart throughout. A DeLong standard error or a bootstrap interval
measures sampling within one split. A `+/-` across seeds measures split variability. For the
headline comparison these agree: +0.0107 with a within-split SE of 0.0040, against +0.0099 +/-
0.0034 across seeds. [V, results/ablation.json]

The gain is not extra model capacity. The text vectorizer saturates at 37,558 features, and
raising its cap from 30,000 to 60,000 to 120,000 gives 0.9458, 0.9455, 0.9455.
[V, results/ablation.json capacity_control]

Removing one structural family at a time from the full 0.9230: enum values are worth the most
(0.9135 without), then entity counts (0.9197), numeric and list features (0.9195), payment method
type (0.9212). Argument key names are neutral (0.9249 without, within noise).
[V, results/ablation.json structural_families_removed]

### 2.2 AppWorld

A second benchmark, coding agents rather than customer service, and no natural-language closing at
all. Labelling needs no regex: false success is `complete_task(status="success")` with
`evaluation.success == False`.

Split is by submission, not by task, because AppWorld tasks recur across submissions and a random
split would place the same task's trace on both sides. This is stricter than the tau2 protocol.

| corpus | endpoints | arguments | both | gap |
|---|---|---|---|---|
| any completion claim, 8,467 tasks, 43 submissions | 0.8112 +/- 0.0275 | 0.8383 +/- 0.0252 | **0.8450 +/- 0.0254** | +0.0363 CI95 [0.0269, 0.0457] |
| explicit status, 1,390 tasks, 13 submissions | 0.7892 +/- 0.0598 | 0.8325 +/- 0.0538 | 0.8093 +/- 0.0581 | +0.0294 CI95 [0.0020, 0.0601] |

Both corpora: 10 of 10 seeds. Precision at a 10% flag rate on the larger corpus rises from 0.9363
to 0.9595. [V, results/appworld.json]

On the 1,390-task corpus, arguments alone score above the combination. We attribute this to sample
size, since the standard deviation there is more than double the larger corpus, and we quote the
8,467-task result. [H]

### 2.3 Prevalence is agent-dependent, not a constant

The reference work reports 75.8% for AppWorld self-assessing trajectories with explicit status
claims [C, arXiv 2606.09863 abstract]. Across the 11 AppWorld submissions with at least 100
explicit success claims, the false-success rate among those claims ranges from 19.5% to 82.9%,
median 65.2%, pooled 55.9%. [V, results/appworld_prevalence.json]

A single prevalence figure for a benchmark is therefore not meaningful. The reference value sits
inside our range and above our pooled figure. The four-fold spread across agents is the more
useful observation, and it means prevalence claims should be reported per agent.

## 3. Robustness, on tau2 only

The reference work reports that rewriting honest-failure closings in confident style flips its
detector on 10 of 50 cases [C, arXiv 2606.09863 Appendix H]. We extend this and reach a different
conclusion about what it means.

We built two rewriters. `swap_lexicon` substitutes the trigger lexicon for equivalent phrasing and
keeps every fact, producing 851 distinct outputs from 851 inputs, removing the trigger from 90.6%
and preserving 0.943 mean character similarity. `assert_completion` excises admission sentences
and prepends a completion claim built from the message's own entity.

Under `swap_lexicon` the text detector falls from 0.9458 to 0.9212, and the combined detector from
0.9557 to 0.9430. The combined detector loses roughly half as much and beats text on 10 of 10
seeds under attack, +0.0194 CI95 [0.0104, 0.0284]. Precision at a 10% flag rate under attack is
0.4818 for text and 0.5805 for combined. [V, results/ablation.json]

Two negative results belong here.

First, the text detector is not a lexical rule. Removing 90.6% of its trigger vocabulary while
keeping the message costs 2.5 AUROC points. Appendix H's 20% flip rate invites the opposite
reading. [V, results/ablation.json]

Second, template-based rewrites are not a valid attack. Replacing a 91-word message with one of
three generic sentences drops the text detector to 0.6728, but that measures distribution shift,
not evasion. We report it only as a caution against the method. [V, results/robustness.json]

`assert_completion` flips the detector on 13.3% +/- 1.5% of 218 substantive honest failures, each
rewritten into a genuine false success with its body intact. The detector misses 86.7% of them.
[V, results/robustness.json]

## 4. Two measurement defects

### 4.1 The reference corpus cannot be reconstructed

The reference work states its corpus is 9,876 tau2-bench trajectories, 8,146 successful and 1,730
failures, obtained from Sierra's public leaderboard [C, arXiv 2606.09863 §3.1]. That composition
does not exist in the leaderboard data.

The leaderboard at the paper's date is `manifest.json` at tau2-bench commit `c7e436f33`. Exactly
12 listed submissions carry airline, retail or telecom trajectories. All 37 of their files return
a `Last-Modified` of 2026-03-23 and none was touched after June, so they are the same bytes the
authors read. No subset of whole submissions matches. An exhaustive meet-in-the-middle over all 37
files returns zero solutions for (8146, 1730). No `termination_reason` slice reaches the required
rate: user_stop 26.5%, agent_stop 27.9%, max_steps 100%. The pooled failure rate is 26.8% against
the required 17.5%. [V, results/corpus_provenance.json]

We do not claim an error in the reported numbers. We claim the corpus is not reconstructible from
the stated source, which makes the result unverifiable by anyone without the authors' data.

### 4.2 tau2-bench rewards auto-pass

tau2-bench issue #384 reports that `evaluator_communicate.py:28` and
`evaluator_nl_assertions.py:38` return reward 1.0 when the criteria field is empty, and that 23
airline and 6 retail tasks are scored on inaction alone [C, tau2-bench issue #384].

In the shipped data, 88.0% of airline simulations carry a COMMUNICATE component with no criteria
to evaluate, contributing 1.0 to the reward breakdown while measuring nothing. 46.0% of airline
simulations are on the flagged task list, and 80.4% of those score full reward against a 68.9%
base rate. Retail is affected less: 31.4% empty gate, 5.3% on the list, 89.2% full reward there
against a 76.6% base. [V, results/corpus_audit.json issue_384_auto_pass]

The effect on detector scores is small. Removing harness control tokens costs the text detector
0.0084 AUROC, from 0.9438 to 0.9354, and costs a tool-names-only detector 0.0000, unchanged at
0.8865. Both figures are seed 42 and predate the argument features.
[V, results/corpus_audit.json ablations]

## 5. What a deployment would face

Sections 2 and 3 measure a detector. This section measures the four things that decide whether one
can be shipped. All four are on tau2 unless stated.

### 5.1 Scoring agents it has never seen

A deployed detector scores agents absent from its training data. Holding out whole submissions,
the ranking survives.

| protocol | text | structural | combined |
|---|---|---|---|
| random split | 0.9458 | 0.9230 | 0.9557 |
| 30% of submissions held out, 10 seeds | 0.9244 +/- 0.0189 | 0.9067 +/- 0.0160 | 0.9347 +/- 0.0142 |
| leave one submission out, 12 folds | 0.9339 +/- 0.0326 | 0.9146 +/- 0.0232 | 0.9447 +/- 0.0243 |

Combined beats text on 11 of 12 held-out submissions. The worst fold is
`claude-3-7-sonnet_anthropic_2024-06-20` at 0.8797, the oldest model and the one with the highest
failure rate at 38.5%. [V, results/heldout.json]

Precision does not survive in the same way. At a 10% flag rate the combined detector gives
0.4861 +/- 0.1087 across held-out submissions, ranging 0.32 to 0.65.

The cause is not instability. Across the twelve folds, the correlation between the held-out agent's
own failure rate and the precision achieved on it is **r = 0.932**. Failure rate ranges 3.3% to
18.8% and precision follows it from 0.2653 to 0.8784. Precision at a fixed flag rate is bounded by
base rate over flag rate, so it moves with the deployment, not with the model.
[V, results/heldout.json]

The consequence for anything shipped: the ranking can be trained once and reused, the threshold
cannot. It has to be fitted against the deployment's own prevalence.

### 5.2 The feature set a real trace actually carries

Our extractor reads fields specific to the benchmark: a `requestor` marking dual-control turns, the
final message role, and a hand-written list of 22 state-changing tool names. OpenTelemetry GenAI
carries none of those. It carries `gen_ai.tool.name`, `gen_ai.tool.call.arguments` and per-call
error status, and every `gen_ai.*` attribute was still Development status in July 2026
[C, OpenTelemetry semantic conventions registry].

Restricting the extractor to those three fields:

| variant | AUROC | P@10% |
|---|---|---|
| benchmark-native | 0.9230 +/- 0.0081 | 0.5150 |
| three OTel fields, caller supplies the tool list | 0.9207 +/- 0.0071 | 0.5091 |
| three OTel fields, no tool list, every call scored | **0.9254 +/- 0.0055** | 0.5208 |

DeLong on the confirmatory split with Holm across three comparisons: native versus zero-config is
+0.0045 at p 0.232, not significant. Native versus caller-supplied-list is significant at p_holm
7.69e-03, and caller-list versus zero-config is significant at p_holm 6.82e-03, both in favour of
using no list. [V, results/portability.json]

Two results follow. The portable feature set matches the benchmark-native one, so nothing is lost
by reading only what a standard trace provides. And the hand-written list of state-changing tools
was making the detector worse: scoring every call, including read-only ones, is better than
filtering to the ones we judged relevant. The most portable configuration is also the best, and it
has nothing to configure.

### 5.3 Cost

Measured on 500 trajectories of median 28 messages, single-threaded CPU, feature extraction
included:

| stage | mean | p50 | p95 | max |
|---|---|---|---|---|
| extract | 0.065 | 0.059 | 0.121 | 0.307 |
| score | 0.395 | 0.379 | 0.507 | 0.873 |
| end to end | 0.460 | 0.438 | 0.627 | 1.180 |

Milliseconds. Batched, 0.124 ms per trajectory. [V, results/deployment.json]

### 5.4 No model serves both benchmarks

A detector trained on tau2 scores AppWorld at 0.4948. Trained on AppWorld it scores tau2 at 0.5213.
Both are chance. The two feature vocabularies share 18 tokens out of 11,729 and 30,000, and all 18
are call-count buckets or incidental value matches such as `val:city=seattle`.
[V, results/deployment.json]

This is not a degenerate comparison. Every one of the 8,467 AppWorld trajectories contains at least
one feature the tau2 vectorizer recognises, so the model produces real scores rather than a
constant. The overlap exists and carries no signal.

The consequence: there is no useful pretrained model for this task. A detector is fitted per
environment, from that environment's own labelled runs.

## 6. Related work

Tool-call arguments are a well-studied source of agent failure. Our contribution is narrower than
that literature and every qualifier below is load-bearing.

**Parameter failure taxonomies.** Butterfly Effects in Toolchains derives five parameter failure
categories and probes them with 15 input perturbations, concluding that parameter name
hallucination stems from model limits while other patterns come from input sources
[C, arXiv 2507.15296]. ToolScan characterises seven tool-use error types including incorrect
argument values, names and types. These works taxonomise causes. They do not build a classifier,
and they do not compare argument features against tool-name features for predictive performance.

**Diagnostic benchmarks.** ToolFailBench measures Tool-Skip, Result-Ignore, Output-Fabrication and
Unnecessary-Tool-Use over 1,000 single-turn tasks with mock returns, using a rule classifier and
two LLM judges [C, arXiv 2607.04686]. It is a benchmark for characterising models, built on
synthetic parametric traps. Ours is a detector over multi-turn traces that real agents produced on
public leaderboards.

**Trace features for silent failure.** Structural trace features, specifically path features over
the sequence of agent, tool and LLM calls, plus token-usage features, have been used to detect
silent failures in multi-agent trajectories [C, arXiv 2511.04032]. This is the closest work on the
feature axis. It uses call sequence and volume, not the values passed to those calls.

**Argument validation in evaluation stacks.** Production eval practice matches argument shape and
values against a gold call. That requires the gold call. Our detector has no reference to compare
against and predicts from the trace alone.

**Where most parameter work differs from ours.** The failures above are largely calls that fail:
malformed JSON, a hallucinated parameter name, a value the API rejects. The false successes we
study are calls that *succeed*. Their arguments are well-formed and accepted, and they are the
wrong values. Butterfly Effects contains one example of this kind, a region set to "US" for a
query about Australia, but treats it as one category among five rather than as the dominant mode.

Stated precisely, our claim is: **tool-call argument features predict task-level failure better
than tool-name or endpoint features, at inference time, without a gold reference, on two
multi-turn agent benchmarks.**

## 7. Limitations

**Scope.** This work is about a claim of task completion contradicted by verifiable environment
state. It says nothing about factual hallucination, reasoning quality, or research agents that
search and then fabricate. Our method requires a programmatic ground truth to diff against. No
such ground truth exists for a research agent writing a summary, and we make no claim there.

**One attack surface.** The adversarial results exist only on tau2-bench. AppWorld has no
natural-language channel to attack.

**No comparison against an LLM judge.** Prior work reports judges at AUROC 0.65 on tau2 and 0.54 on
AppWorld [C, arXiv 2606.09863]. We did not reproduce that and make no claim about judges. Every
judge number in this area that we might cite is someone else's measurement.

**Content capture.** The portable feature set depends on `gen_ai.tool.call.arguments`, which
OpenTelemetry records only when privacy policy permits. A deployment with content capture disabled
has nothing for this method to read.

**Convention stability.** Every `gen_ai.*` attribute was Development status as of July 2026, with
no stability guarantee on names [C, OpenTelemetry semantic conventions registry].

**Attacker strength.** Both rewriters are deterministic and rule-based. A model-generated rewrite
would be more fluent and is untested. Our results are therefore a floor on the text detector's
fragility, not an estimate of it. [H]

**Label quality.** Every AppWorld and tau2 number is scored against labels with the defects in §4.

**Corpus.** Our tau2 corpus is 12,859 trajectories against the reference 9,876, and is not the
same sample. Direct numeric comparison with the reference AUROC is not valid.

**Licence.** AppWorld experiment outputs are Apache 2.0 with an added requirement that public
redistribution of them or their derivatives be in encrypted form. Extracted trajectories cannot be
shipped.

## 8. Reproduction

Corpus pin, which the reference work does not have:

- tau2: S3 bucket `sierra-tau-bench-public`, keys in `data/tau2/_bucket_keys.json`, all trajectory
  files `Last-Modified 2026-03-23`, leaderboard membership at tau2-bench commit `c7e436f33`
- AppWorld: `StonyBrookNLP/appworld-leaderboard`, 48 bundles under git-lfs, unpacked with
  `appworld` 0.1.3.post1

Environment: python 3.12.3, scikit-learn 1.5.2, numpy 2.5.3. Every table above is produced by one
committed script writing one JSON file under `results/`.

## 9. Scripts

| section | script | results file |
|---|---|---|
| §1 mechanism | `scripts/mechanism.py` | `results/mechanism.json` |
| §2.1 tau2 detectors | `scripts/ablation.py`, `scripts/detectors.py` | `results/ablation.json`, `results/detectors.json` |
| §2.2 AppWorld | `scripts/appworld_detectors.py` | `results/appworld.json` |
| §2.3 prevalence | `scripts/appworld_prevalence.py` | `results/appworld_prevalence.json` |
| §3 robustness | `scripts/robustness.py`, `scripts/adversarial.py` | `results/robustness.json`, `results/adversarial.json` |
| §4.1 corpus | `scripts/corpus_provenance.py --check-timestamps` | `results/corpus_provenance.json` |
| §4.2 auto-pass | `scripts/corpus_audit.py` | `results/corpus_audit.json` |
| §5.1 held out | `scripts/heldout.py` | `results/heldout.json` |
| §5.2 portability | `scripts/portability.py` | `results/portability.json` |
| §5.3 and §5.4 | `scripts/deployment.py` | `results/deployment.json` |

Data acquisition: `scripts/fetch_tau2.py` for tau2, and for AppWorld a git-lfs clone of
`StonyBrookNLP/appworld-leaderboard` unpacked with `appworld unpack`.

## 10. Open

- Does the finding hold with a model-generated attacker rather than rule-based rewriters?
- Does it hold on a third benchmark with a different action representation?
- The abstract form of the result is that fine-grained action content predicts failure better than
  coarse action shape. Whether that transfers to agents whose actions are searches rather than API
  calls is untested and is a separate project. [H]
