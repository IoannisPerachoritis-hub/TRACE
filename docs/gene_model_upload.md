# Uploading a custom gene model

TRACE annotates LD blocks and significant SNPs with overlapping and flanking
genes. The tomato gene model is bundled; for any other species — or a different
assembly/build — supply your own gene-coordinate table on the
**Post-GWAS Analysis → Gene Annotation** tab (or the pipeline's *"Gene coordinates CSV"*
uploader).

## Required format

A CSV/TSV with one row per gene. Header names are case-insensitive and several
aliases are accepted:

| meaning     | accepted column names                     | required                     |
|-------------|-------------------------------------------|------------------------------|
| chromosome  | `Chr`, `CHROM`, `chr`, `chromosome`       | yes                          |
| gene start  | `Start`, `START`, `start_pos`, `begin`    | yes                          |
| gene end    | `End`, `END`, `end_pos`, `stop`           | yes                          |
| strand      | `Strand`, `STRAND`                        | optional (defaults to `.`)   |
| gene ID     | `Gene_ID`, `GENE`, `name`, `gene_name`    | yes                          |

Coordinates are 1-based gene start/end in base pairs. A leading unnamed index
column (as `pandas.DataFrame.to_csv` writes by default) is tolerated.

## The one rule that matters: same assembly as your VCF

Gene coordinates **must be in the same genome assembly as the SNP positions in
your VCF.** If they are not, every gene name and distance TRACE reports is wrong
by the offset between the two assemblies — silently. (This is exactly the trap
behind the bundled tomato tables: the Varitome SNPs are SL2.5, while the bundled
gene models are SL3.1 / SL4.0, which sit ~0.5 Mb and more away — so bundled
tomato annotation is positional, not coordinate-exact.)

After you upload, TRACE shows a per-chromosome table of the number of genes and
the coordinate range (`min_start`, `max_end`, `span_Mb`). **Check that these
ranges cover the same span as your SNP positions.** A gene model whose
chromosome 1 ends at ~90 Mb against SNPs that run to ~98 Mb (or vice versa) is a
different build — do not use it.

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
`canon_chr`, but a mismatch there drops annotations for the affected
chromosomes — which the per-chromosome upload summary will make visible (a
chromosome present in your VCF but missing from the summary table).
