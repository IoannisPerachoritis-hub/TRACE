# Input files

TRACE takes a VCF of genotypes and a table of phenotypes. A gene model for the annotation and a
table of covariates are optional.

## Genotypes: a VCF

TRACE reads genotypes from a VCF only, plain (`.vcf`) or compressed (`.vcf.gz`). Biallelic SNPs
work best. TRACE stores each call as the count of the alternative allele (0, 1 or 2 for a diploid)
and fills missing calls by mean imputation (the default) or by LD-kNNi (`--impute ldknni`). LD is
computed from the analysed genotypes.

An imputed VCF can carry a quality field per SNP (INFO, DR2, R2 or AR2). TRACE finds it by itself,
and `--info-thresh` drops the SNPs below a value.

Chromosome names lose their common prefixes (`chr`, `SL4.0ch`, `Ca`, `Os`, `Gm`), and the names
that then read as positive whole numbers are kept. The number of chromosomes comes from the data.
A name that does not read as a number (an unplaced scaffold, for example) is set aside as "ALT"
and left out of the LOCO kernels and the LD pruning.

### PLINK, CSV and Zarr files

TRACE does not read PLINK files (`.bed`, `.bim`, `.fam`), CSV genotype tables or Zarr stores.
Convert PLINK files to a VCF first. PLINK 2 does it in one command. On the PLINK files of the
paper's rice VCF it gives that VCF again, identical apart from the date in its header:

```bash
plink2 --bfile mydata --chr-set 12 --fa genome.fa --ref-from-fa --export vcf bgz id-paste=iid --out mydata
```

This writes `mydata.vcf.gz`. In PLINK 2's `--chr-set 12`, put your species' number of chromosomes
(12 for tomato or rice). PLINK 2's `--fa` with `--ref-from-fa` takes the reference allele of each
SNP from the reference genome, because a `.bed` file does not record which allele is the
reference. Without a reference genome, leave out both of these PLINK options. The VCF then names as
reference the allele PLINK 2 takes for it from the `.bim` file, which can flip the sign of an
effect but changes no p-value. PLINK's `id-paste=iid` writes each sample's ID once, so the VCF's
sample names match the phenotype file.

## Phenotypes: a table

A CSV or TSV file with one row per sample: a column of sample IDs that match the VCF, then one
column per trait. Traits are numeric. A rank-based inverse normal transform (`--norm int`) helps
with a skewed trait.

The CLI takes the first column as the sample IDs. The GUI takes the first column named
accessions, accession, sample, samples, iid, genotype, id, sample_id or sampleid (in any case),
else the first column. Put the sample IDs in the first column and both read the file the same way.

## Covariates (optional)

Covariates go in their own table (`--covar`, or the GWAS page's covariate upload): the sample IDs
first, then numeric columns. The GUI can also take covariates from columns of the phenotype file.
The CLI does the same with `--covar pheno.csv --covar-cols weight_g`.

## Gene models (optional)

The tomato gene models are bundled: SL3.1 (NCBI RefSeq, GCF_000188115.5, the default) and
ITAG4.0 (`--genome-build SL4`). Gene models and SNP positions must come from the same genome
build. The Varitome VCF uses SL2.5 positions, which match neither bundled model, so its annotation
is positional only.

For any other species, give a gene table or a GFF3 file (`--species custom --gene-model FILE`,
with descriptions in `--gene-desc FILE`), or upload them in the Post-GWAS Analysis page's Gene
Annotation tab. [Gene model upload](gene_model_upload.md) gives the format.

## The QC data of the paper's benchmarks

The benchmarks start from TRACE's own QC of the public Varitome VCF. `--export-qc` writes that QC
data (the genotype matrix, the SNP map and the phenotype) into the results ZIP, and
`benchmarks/make_qc_data.sh` runs TRACE with it and puts the files where the benchmark scripts
look. See [the benchmarks README](../benchmarks/README.md).
