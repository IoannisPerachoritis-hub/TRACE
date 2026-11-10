# Uploading a custom gene model

TRACE annotates each LD block and each unblocked significant SNP with the genes in its
candidate interval: from the marker before the first SNP to the marker after the last.
The tomato gene model is bundled; for any other species, or a different assembly/build,
supply your own gene file in **Step 1: Gene models** of the **Post-GWAS Analysis → Gene
Annotation** tab, in the GWAS page's One-Click *"Override gene files"*, or on the command
line (below). The "?" next to each uploader gives the same rules as this page.

## Required format

Plain text with one row per gene and a header row: comma-, semicolon- or tab-separated,
UTF-8 (Excel's "CSV UTF-8" is fine), not compressed. Header names are case-insensitive,
several aliases are accepted, and other columns are ignored. Use one column for each
field: a file with two columns for one field (for example both `Gene_ID` and `Name`) is
refused with a message naming them.

| meaning     | accepted column names                     | required                     |
|-------------|-------------------------------------------|------------------------------|
| chromosome  | `Chr`, `CHROM`, `chr`, `chromosome`       | yes                          |
| gene start  | `Start`, `START`, `start_pos`, `begin`    | yes                          |
| gene end    | `End`, `END`, `end_pos`, `stop`           | yes                          |
| strand      | `Strand`, `STRAND`                        | optional (defaults to `.`)   |
| gene ID     | `Gene_ID`, `GENE`, `name`, `gene_name`    | yes                          |

Coordinates are 1-based gene start/end in base pairs; rows whose start or end is not a
number are skipped. A leading unnamed index column (as `pandas.DataFrame.to_csv` writes
by default) is tolerated. Example, a row of TRACE's SL3.1 table:

    Chr,Start,End,Strand,Gene_ID
    1,30726,43941,-,Solyc01g005020.3

## Descriptions file (optional)

One gene per line: the gene ID, a TAB, then the description. No header is needed; lines
starting with `#` are ignored. Lines without a TAB are skipped, so a comma-separated file
adds no descriptions. The IDs must match the gene file's; version endings are ignored
(`Solyc01g005020.3` matches `Solyc01g005020`). Example, from TRACE's SL3.1 descriptions
(the gap is one TAB):

    Solyc01g005020.3	Salt Overly Sensitive 1

## Command line

    trace-gwas --vcf genotypes.vcf.gz --pheno phenotypes.csv --trait my_trait \
        --species custom --gene-model gene_model.csv --gene-desc descriptions.txt

`--gene-model` is required with `--species custom`; `--gene-desc` is optional (without it
a custom run lists gene IDs only). The run's `run_metadata.json` names both files.

## The one rule that matters: same assembly as your VCF

Gene coordinates **must be in the same genome assembly as the SNP positions in
your VCF.** If they are not, every gene name and distance TRACE reports is wrong
by the offset between the two assemblies, silently. (This is exactly the trap
behind the bundled tomato tables: the Varitome SNPs are SL2.5, while the bundled
gene models are SL3.1 / SL4.0, which sit ~0.5 Mb and more away, so bundled
tomato annotation is positional, not coordinate-exact.)

After you upload, TRACE shows a per-chromosome table of the number of genes and
the coordinate range (`min_start`, `max_end`, `span_Mb`). **Check that these
ranges cover the same span as your SNP positions.** A gene model whose
chromosome 1 ends at ~90 Mb against SNPs that run to ~98 Mb (or vice versa) is a
different build. Do not use it.

## Deriving a gene model from a GFF3

```python
import pandas as pd

cols = ["seqid", "source", "type", "start", "end",
        "score", "strand", "phase", "attributes"]
gff = pd.read_csv("your_genome.gff3", sep="\t", comment="#",
                  names=cols, dtype=str)

genes = gff[gff["type"] == "gene"].copy()
genes["Gene_ID"] = genes["attributes"].str.extract(r"ID=([^;]+)")

out = genes.rename(columns={"seqid": "Chr", "start": "Start",
                            "end": "End", "strand": "Strand"})
out[["Chr", "Start", "End", "Strand", "Gene_ID"]].to_csv(
    "gene_model.csv", index=False)
```

Make sure the `Chr` values match how chromosomes are named in your VCF (e.g.
`1` vs `chr1` vs `SL2.50ch01`). TRACE canonicalises common Solanaceae naming via
`canon_chr` (`chr01`, `Chr1`, `SL2.50ch01` and `1` are all chromosome 1), but a mismatch
there drops annotations for the affected chromosomes. TRACE then warns, after the upload:
"No genes for chromosome(s) … of your genotype data; check that the chromosome names
match." The per-chromosome upload summary shows the same gap.
