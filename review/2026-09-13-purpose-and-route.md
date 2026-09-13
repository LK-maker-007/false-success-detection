# Review: project purpose and route

Date: 2026-09-13. Read-only, nothing modified. Delivered across several messages; consolidated here.

### 1. Problem real, framing wrong
Practitioner pain is well documented but lives in **coding and GUI agents**, not customer service:
GitHub issues against Claude Code, Codex, Copilot and Cursor; METR; VLAA-GUI reporting over 86%
false completion on OSWorld.

"Roughly half of all failures" is wrong. By the repo's own labels it is **24.7% on tau2** and 31.2%
on AppWorld. Airline and retail, the two domains producing the large numbers, are exactly the two
with the open reward auto-pass bug. **Telecom, the only domain whose reward function works, has 14
false successes out of 3,994.**

### 2. The route has one fatal flaw, and it is not the deployment constraints
The constraints (supervised, no shippable weights, local threshold fitting, content capture
required) are survivable. The central claim is not: arguments beat tool names by +0.0075 as
published, and by **-0.0181 on tau2 and -0.0317 on AppWorld** once tasks are held out.

### 3. Strongest dead-end argument
The classifier approximates an oracle you must already own in order to build it, and roughly 0.83
AUROC of the published task is recoverable from the labelling regex alone.

### 4. Highest-value next move
Publish the evaluator-validity audit. Specifically: the widely repeated finding that **dual control
suppresses false success tenfold is confounded** with telecom being the only domain whose reward
function works. Data, scripts and half the prose are already in the repo.

Venue note: the NeurIPS 2026 "Who Verifies the Agents?" workshop closed submissions 2026-08-29.
Remaining routes are arXiv plus the tau2-bench issue tracker directly (#384 has zero maintainer
comments in ten weeks; a reproduction with numbers attached is the fastest way to get it acted on),
or an ICLR/ACL workshop cycle.

### 5. Nobody is doing this better, but eleven vendors ship the feature name
None verifies against environment state. None publishes an accuracy number. None ships weights.

Useful framing for a related-work table, three tiers:
- **A, judge over transcript.** Azure `task_adherence.prompty` (MIT, `azure-ai-evaluation` 1.18.5)
  is a free zero-shot false-success judge whose rubric names the failure mode verbatim: "Claims
  external actions or results critical to the objective with no corroborating evidence in the tool
  interactions"; "Claims external actions or data without matching TOOL_CALLS; TOOL_CALLS are the
  source of truth"; "Mishandles asynchronous operations (e.g., claims completion when only
  queued)". Datadog ships `Tool Argument Correctness` and `Goal Completeness` as judge templates,
  the nearest commercial analogue to this project's specific feature claim.
- **B, match against a supplied label.** Future AGI `fi.evals.TaskCompletion`, published scoring
  decomposition (20% trajectory final-step, 50% outcome match, 30% success criteria), requires
  `expected_result` plus a `success_criteria` list.
- **C, genuinely environment-grounded.** One commercial product (Coval, voice QA, pre-production)
  and a long tail of 0-4 star repos.

The draft says "No comparison against an LLM judge" and cites someone else's 0.65. Running Azure's
prompt over the tau2 corpus is cheap and closes the most obvious reviewer gap. If it beats 0.65 by a
wide margin, that is itself a finding about how the judge baseline in this literature was built.

### 6. Ambition
Not reachable via this detector. Reachable via the audit.
