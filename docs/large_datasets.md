# Large datasets: keeping the genotype table on disk

TRACE holds one table of genotype calls (samples x SNPs) for the whole run. Held in memory,
that table is copied several times by the analysis steps, so the memory a run needs grows with
samples x SNPs: about 50 bytes per genotype call at the peak. At 1,000 samples and 100,000 SNPs
that is about 5 GB; at 1,000 samples and 1 million SNPs about 50 GB.

With the table **on disk**, the VCF is read in pieces into a file of one byte per genotype call,
and every step reads from that file only the columns it works on. Memory then no longer grows
with samples x SNPs. A simulated panel of 996 samples and 1,070,795 SNPs ran on a 16 GB desktop
in 39 minutes at 2.1 GB of memory (MLM; 75 minutes and 2.6 GB with MLM, MLMM and FarmCPU).

**The results are the same either way.** Every step receives the same numbers, in the same
number format and order, as it does from the table in memory, so the results files are
byte-identical at a given thread count (see "Reproducibility" below). You can check this
yourself (see "Checking that both modes agree").

## Choosing where the table is kept

`--geno-store` takes three values:

| Value | What TRACE does |
|---|---|
| `auto` (default) | Before reading the genotypes, TRACE estimates the memory an in-memory run would need and compares it with the free memory. If the estimate is at most 80% of the free memory, the table stays in memory; otherwise it goes on disk. |
| `memory` | The table stays in memory. If the estimate is above the free memory, the run stops before reading, with a message naming both numbers. |
| `disk` | The table goes on disk. |

The log says which mode ran and why, in one line, for example:

```text
This dataset has about 1,066,511,820 genotype calls; an in-memory run would need about 54.3 GB,
and 5.9 GB are free, so TRACE keeps the genotype table on disk. Results are the same either way.
```

(a simulated panel of 996 samples and 1,070,795 SNPs on a 16 GB desktop).

The same line, the estimate and the free memory are recorded in `run_metadata.json` in the
results ZIP (keys `Genotype table`, `Genotype table choice`, `Genotype table reason`,
`Memory estimate, in-memory run (GB)`, `Available memory at start (GB)` and
`Genotype calls (estimate)`). These keys are the only difference between the ZIPs of the two
modes; the HTML report is the same.

### How the estimate is made

TRACE reads the number of samples from the VCF header. For the number of SNPs it takes the number
of records: from the file size for a plain or a BGZF-compressed VCF (an upper bound, about twice
the real count for a file with only the GT field, more for files with other FORMAT fields), or
by counting the records in one pass for other gzip files (11 seconds for the 99 MB file of that
example). Before choosing the disk, or stopping a
run, a bound is replaced by an exact count, so no run goes to disk or stops on a loose bound.

The estimate is the memory peak of an in-memory run: a fixed part (800 MiB with MLM, 1,200 MiB with
MLMM or FarmCPU) plus 50 bytes per genotype call, plus more for subsampling with several workers
(300 MiB and 12 bytes per call for each extra worker), `--export-qc` (24 bytes per call) and LD-kNNi
(16 bytes per call and 24 KiB per SNP), plus the n x n kinship matrices. It was fitted to measured peaks of
in-memory runs and lies at least 10% above each of them.

The free memory is the operating system's available memory. Inside a Linux container with a
memory limit (for example `docker run --memory 4g`), it is what the limit leaves.

## Where the table file goes, and disk space

The table file needs about one byte per genotype call: 1.1 GB for 1,000 samples and 1 million
SNPs. While a plain-text VCF is read, a cleaned copy of it is kept next to the table file, so the
disk also needs the VCF's size once more for that time. TRACE checks the free space on that drive
before reading and stops with a clear message if it is short.

Each run makes its own new folder for the file, named `_trace_geno_store_<process>_<random>`:

* inside the `--output` folder by default (a folder you can already write to, also under Docker
  with the README's bind mount and user setting);
* inside `--geno-store-dir` when you give it, for example a faster or larger drive. The folder is
  created if it is missing; if it cannot be created or written to, the run stops before reading.

The run's folder is removed when the run ends, also when it stops early with an error. A run that
is killed (power cut, out of memory, closed terminal) cannot remove it: the next run in the same
folder names it in the log with its size, and you can delete it. TRACE never deletes a folder it
did not make in this run, since another run may still be using it.

## What is not available with the table on disk

| Option | Why | What happens |
|---|---|---|
| `--impute ldknni` | LD-kNNi ranks every SNP against all others, which needs the whole table in memory | With `--geno-store disk`, the run stops before reading. With `auto`, it stops when the table does not fit in memory, naming the memory needed and the free memory. Use `--impute mean` (the default) for large datasets. |
| `--export-qc` above 100 million genotype calls | it writes the whole table twice as text into the results ZIP | the run stops before reading; below that size it works as usual |
| The QC report with 16.8 million markers or more | its sample concordance counts are exact only below 2^24 markers | the QC report is skipped with a warning; the rest of the run is unchanged |

## How long large runs take

With the table on disk, the steps read the table file many times (51 GB of reads for the three
models at 996 x 1,070,795), but in pieces of at most 8,192 SNPs, so the time is about that of the
in-memory steps. Two steps grow with the number of SNPs in ways worth knowing:

* **LD-block detection** works around every seed SNP (the significant SNPs, at least the top
  `--ld-top-n`): at 996 x 1,070,795 it took 23 minutes for 1,073 seed SNPs. `--ld-seed-mode`
  and a lower `--ld-top-n` give fewer seeds.
* **Per-SNP box plots** take about 5 minutes at that size; `--snp-plots none` skips them.

MLMM scans the whole genome once per forward step, so it takes several times as long as MLM.
Subsampling with the table on disk runs its replicates one after another, with the results of a
parallel run (each replicate uses the thread count a parallel worker would have).

## Reproducibility

The results are byte-identical between the two modes at the same number of threads of the
linear-algebra library (OpenBLAS or MKL). Between thread counts, the last digits of some p-values
change (by about 1 part in 10 million; no significance call changed in our tests), and with
`--boot-jobs` above 1 the subsampling results depend on the threads of each worker: the thread
count you set (for example `OPENBLAS_NUM_THREADS`) or, if you set none, cores / workers. Every
run records the library and its version, the thread count, the cores, `--boot-jobs` and the
threads per worker in `run_metadata.json` and in the report's provenance table.
To reproduce a run exactly, set the same thread count, for example `OPENBLAS_NUM_THREADS=4`.

## Docker

The default folder for the table file is inside `--output`, so the README's bind mount (and
`--user "$(id -u):$(id -g)"` on Linux) gives TRACE a folder it can write. A Docker named volume
works if the user TRACE runs as can write to it; a new volume is owned by root, so TRACE's image
user cannot write there unless you give the volume to that user. A container memory limit is read
by `auto`: a run that does not fit under the limit keeps its table on disk.

## Checking that both modes agree

Run the same analysis twice, once with each mode, into two folders, and compare the two ZIPs:

```bash
trace-gwas --vcf my.vcf.gz --pheno pheno.csv --trait Yield --output run_memory/ --geno-store memory
trace-gwas --vcf my.vcf.gz --pheno pheno.csv --trait Yield --output run_disk/ --geno-store disk
python scripts/compare_result_zips.py run_memory/GWAS_*.zip run_disk/GWAS_*.zip
```

The script compares every file of the two ZIPs byte for byte, leaving out only the
`run_metadata.json` keys above and the report's "Generated" time.
