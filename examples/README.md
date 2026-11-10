# Example dataset

This directory holds a small, byte-reproducible synthetic dataset for verifying a
TRACE installation end-to-end in about 7 seconds:

- `example.vcf.gz` — 100 inbred lines and 600 SNPs on 3 chromosomes. The lines are
  mosaics of four founder haplotypes, so neighbouring SNPs are in linkage
  disequilibrium (LD) and LD decays with distance.
- `example_pheno.csv` — `Trait1` carries one QTL, `chr02_1010000` (h² ≈ 0.3), inside
  an LD block on chromosome 2; `Trait2` is noise.
- `simulate_example.py` — regenerates both deterministically (seed 42)
- `run_example.sh` — one-command end-to-end run

At TRACE's defaults, Trait1's significant SNPs all lie around the QTL, where TRACE
finds one LD block, tests its haplotypes and draws its LD heatmap, so every Post-GWAS
tab has something to show.

**The full walkthrough — command line and web app — lives in
[`docs/tutorial.md`](../docs/tutorial.md).**

Quick start:

```bash
bash examples/run_example.sh
```
