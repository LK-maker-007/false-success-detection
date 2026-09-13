# Review: reproducibility and CI

Date: 2026-09-13. Read-only, nothing modified.

**Verdict:** the arithmetic is honest, the inputs are not addressable. A stranger with a laptop can
rebuild the tau2 half approximately, and cannot rebuild the AppWorld half at all. Nothing in the
repo would tell them if they had rebuilt the wrong corpus.

## Blocks reproduction

### 1. The AppWorld corpus has no acquisition path, no pin, and no environment
Affects `results/appworld.json`, `results/appworld_prevalence.json`, the `cross_benchmark` block of
`results/deployment.json`, and paper sections 2.2, 2.3, 5.4.

- The clone on disk is at `c1f56015cf7c3441ff1933f5bfbec879798d7bbe`. That SHA appears in no
  tracked file. Upstream keeps merging submissions, so `git clone` today yields a different corpus.
  The counts 8,467 / 43 submissions and 1,390 / 13 are not recoverable by re-cloning.
- `fsdetect/appworld.py:68` raises "run appworld unpack first". Nothing says what `appworld` is,
  what version, or how to install it. The only mention of 0.1.3.post1 is prose at draft.md:361.
- The unpack environment `.venv-appworld/` is gitignored by `.gitignore:6` and never exported.
  `appworld` is not importable from `.venv`, which is what CI and every script use.
- Counts do not reconcile: draft says 48 bundles; disk has 50 entries; 45 have `evaluations/`;
  43 contribute records.
- The AppWorld licence forbids unencrypted redistribution, so a script is the only possible path
  and it does not exist.

### 2. The tau2 corpus is not pinned, and the stated pin is gitignored
- draft.md:358 names `data/tau2/_bucket_keys.json` as the pin. `.gitignore:21` excludes it.
- `scripts/fetch_tau2.py:18-19` selects by regex against a live S3 listing. A new Sierra submission
  enters the match set automatically and every tau2 number changes.
- `fetch_tau2.py:73-75` verifies only that bytes written equal bytes the same listing advertised.
  Cannot detect a replaced file of equal size, cannot detect a changed match set.
- `MANIFEST_COMMIT = "c7e436f33"` is emitted into JSON and never checked. An assertion.
- Partial mitigation: `results/corpus_provenance.json` carries a usable fingerprint, but no script
  verifies against it.

### 3. CI verifies zero numbers
`.github/workflows/ci.yml` runs ruff, pytest, and `--help` per script. It never touches `data/` and
never re-derives a single value in `results/`. The `--help` loop is not a smoke test; argparse exits
before any code path runs.

### 4. CI does not install the lock; the lock is an orphan
- CI runs `pip install -e .[dev]`. pyproject pins scikit-learn but leaves numpy open and scipy
  unconstrained. Results files record numpy 2.5.3, CI resolves whatever is current.
- `requirements-lock.txt` is referenced by nothing. No hashes. No README, no bootstrap instruction.
- Version skew: `OptimizeWarning: Unknown solver options: iprint` on every fit. sklearn 1.5.2 runs
  outside its tested scipy range.

## Serious, not blocking

5. **A superseded AppWorld run is silently in the corpus.**
   `lara_test_normal_OLD_54.8_hardcoded/` contributes 165 of 8,467 records alongside
   `lara_test_normal` (166). Upstream named it OLD and hardcoded. GroupShuffleSplit treats them as
   independent groups.
6. **The section 2.2 split rationale is wrong**, and the code comment repeats it. Grouping by
   submission does not stop task leakage, it guarantees it. Task disjointness requires grouping on
   `task_id`.
7. **Three numbers in section 3 have no producing script**: "851 distinct outputs from 851 inputs",
   "removing the trigger from 90.6%", "0.943 mean character similarity".
8. `heldout.py:120` hardcodes ablation.json's means as `random_split_reference`.
9. Provenance blocks inconsistent; two are near-empty. No results file records a git commit, a data
   checksum, or a timestamp. The latency table records no CPU model.
10. Order-dependent dedup in `corpus_audit.py`: the shared `seen` set iterates FS, HF, AMB, TS, so a
    TS row matching an HF or AMB closing is dropped.
11. `fsdetect/adapters.py`, `fsdetect/api.py`, `tests/test_api.py` untracked. Local pytest collects
    50; a clean checkout collects 29. CI has never run 21 of them.

## Untidy
12. `fsdetect/detect.py:22` cites `results/feature_ablation.json`, which does not exist.
13. `results/adversarial.json` is superseded by robustness.json but still committed and cited.
14. draft.md:217-218 attributes a group-shuffle number to LOSO.
15. `heldout.py:35,37` hardcode `seed=0` for the classifier regardless of the loop seed.
16. `.gitignore` ignores the directory `data/`, so `--out` elsewhere drops 958 MB into a tracked path.

## Clean
No credentials, tokens, or absolute paths in the tracked tree or in `git log -p --all`. One piece of
personal data: a personal Gmail in `pyproject.toml:10`. `.gitignore` covers the real hazards.

`scripts/detectors.py` re-run produced a file byte-identical to `results/detectors.json`.
