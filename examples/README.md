# Example dataset

This directory holds a small, byte-reproducible synthetic dataset for verifying a
TRACE installation end-to-end in under 30 seconds:

- `example.vcf.gz` — ~50 samples, ~510 SNPs, one embedded QTL on chromosome 2
- `example_pheno.csv` — `Trait1` (carries the QTL) and `Trait2` (noise)
- `simulate_example.py` — regenerates both deterministically (seed 42)
- `run_example.sh` — one-command end-to-end run

**The full walkthrough — command line and web app — lives in
[`docs/tutorial.md`](../docs/tutorial.md).**

Quick start:

```bash
bash examples/run_example.sh
```
