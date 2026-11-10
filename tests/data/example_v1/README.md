# The first example dataset (frozen)

`example.vcf.gz` and `example_pheno.csv` are the example `examples/` held before the
current one: 50 samples and 510 SNPs on 3 chromosomes, each SNP drawn independently (so
no linkage disequilibrium), with one QTL at `chr02_860000` (Trait1; Trait2 is noise).
The `examples/simulate_example.py` of that time wrote them (seed 42). They are kept here
byte for byte for the tests that pin their numbers. The example in `examples/` now forms
LD blocks; see its README.
