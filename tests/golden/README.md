# Golden fixtures (Batch A regression harness)

Two-tier golden suite that proves no published number moved during the revision.
**How to know published numbers did not move:** Tier A pins the *algorithm* in CI;
Tier B pins the *published block table* opt-in.

## Tiers

- **Tier A — algorithmic pin, runs in CI, synthetic input.** Six committed
  cases (`blocks_*`), each an engineered LD architecture that drives one branch
  of the detector. Any edit to connectivity, contiguity refinement, gap
  splitting, IoU merging or the min-SNP gate changes a Tier-A golden and fails
  `tests/test_golden_blocks.py`. Self-contained (`≤ 40 kB` per case, `≤ 250 kB`
  total) — no external data.
- **Tier B — manuscript pin, opt-in, real input.** Pins the published Table S8
  block table (`varitome_locule/`) and the tomato locule downstream tables
  (`tomato_locule/`, via `benchmarks/capture_golden.py`). Needs the Varitome QC
  inputs, which are **not** committed; the consuming tests are marked `golden` /
  `manuscript` and skipped otherwise, so a clean clone stays green. These
  directories are populated in Batch B once T-61 reports the flank that
  reproduces Table S8 — do **not** author them from the PDF.

## Layout

```
tests/golden/
  _cases.py            Tier-A input architectures (deterministic from a seed)
  _canon.py            canonical block-table serialisation + GOLDEN_LOCK digest
  regenerate.py        guarded, deliberate regeneration driver
  GOLDEN_LOCK          sha256 over every expected_*.csv (the review tripwire)
  REGENERATION_LOG.md  append-only; one entry per deliberate regeneration
  blocks_*/            Tier-A cases: input.npz + expected_blocks*.csv + meta.json
```

Each `meta.json` records the exact `params` the golden was captured under; the
test reads them back and passes them to the detector, so a golden can never be
silently reinterpreted under different parameters.

## Regenerating a golden (deliberate only)

Behavioural goldens must never be edited by hand. To change one on purpose:

```bash
TRACE_GOLDEN_REGEN=1 python tests/golden/regenerate.py \
    --case blocks_dense_ld \
    --reason "<>=30 chars explaining exactly why the pinned behaviour changed>" \
    --i-am-changing-pinned-behaviour
```

The driver refuses without all of the flags above, refuses under pytest, and
refuses on a dirty tree or detached HEAD (so the recorded `trace_commit` is
meaningful). It rewrites `expected_*.csv`, `meta.json` and `GOLDEN_LOCK`, and
appends one `REGENERATION_LOG.md` entry — the artefacts a reviewer reads. A
regeneration is therefore always a one-line `GOLDEN_LOCK` diff, never hidden
inside a large fixture diff.

## Running locally

```bash
pytest tests/test_golden_blocks.py tests/test_golden_lock.py tests/test_golden_harness.py   # Tier A + guards (CI)
pytest -m golden        # Tier-B real-data pins (needs benchmarks/qc_data/)
pytest -m manuscript    # Tier-B Table S8 pin (needs TRACE_VARITOME_DIR)
```
