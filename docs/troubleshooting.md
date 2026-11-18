# Troubleshooting

This page covers the problems people meet most often: memory, big files, Docker folders and a
Run that stops part way. For large datasets in depth, see [Large datasets](large_datasets.md).

## The run needs more memory than the computer has

TRACE keeps one table of genotype calls for the whole run. In memory it needs about 50 bytes per
genotype call at its peak, so 1,000 samples and 1 million SNPs need about 50 GB. With the table on
disk, memory no longer grows with samples x SNPs, and the results are the same.

By default TRACE decides for you. Before it reads the genotypes, it estimates the memory an
in-memory run needs and compares it with the free memory. When the run does not fit, it keeps the
table on disk and says so in one log line. To choose yourself, use `--geno-store disk` (or
`memory`, or `auto`, the default) in the CLI. In the GUI, open the collapsed section "Large
datasets" on the GWAS page and pick Automatic, In memory or On disk.

With In memory chosen, a run that does not fit stops before reading and names both numbers.
LD-kNNi imputation (`--impute ldknni`) needs the whole table in memory, so for a large dataset use
mean imputation, the default.

## The file is too big to upload

The GUI's upload box takes files up to 2 GB. An upload is held in memory while it arrives, so a
bigger VCF goes through the box under it, "Or type the VCF's location on this computer". Type the
full location of the `.vcf` or `.vcf.gz` file. TRACE reads the file where it is and copies nothing.

The location is a location on the computer that runs TRACE. In Docker it is a location inside the
container: the folder you mounted, for example `/data/panel.vcf.gz` when you mounted your folder at
`/data`. On a shared server the box can be hidden with `TRACE_DISABLE_PATH_INPUT=1`.

## Docker: TRACE cannot write to a named volume

The image runs TRACE as the user `appuser` (user ID 1000). A new Docker named volume belongs to
root, so that user cannot write there. Give the volume to user 1000 once, then use it:

```bash
docker volume create trace_work
docker run --rm --user root -v trace_work:/work --entrypoint chown ghcr.io/ioannisperachoritis-hub/trace:latest 1000:1000 /work
docker run -p 8501:8501 -v trace_work:/work -e TRACE_GENO_STORE_DIR=/work ghcr.io/ioannisperachoritis-hub/trace:latest
```

The GUI then keeps its saved uploads and its table files in the volume, and they stay when the
container is removed. A bind mount of a folder of your own works too. The README's command for
the CLI uses one.

A container with a memory limit (for example `docker run --memory 4g`) gives TRACE only what the
limit leaves, and TRACE's choice of memory or disk follows that.

## A Run stopped part way

A Run stops part way after an error in one step, or when you click Stop at the top right of the
page or open another page while it runs. The steps before the stop have finished and the ones
after it have not, and the session can still hold the tables of the finished steps. Those results
are partial: they lack the later models, blocks, tests or the report.

1. Read the message on the page, or the last lines of the CLI's log. It names the step that
   failed.
2. Fix the cause, for example the memory (see above) or a trait with too few samples.
3. Run again until the run finishes. The GUI then shows the results summary and offers the
   results ZIP. The CLI writes its ZIP and exits with code 0.
4. To start clean, click Clear session data on the GWAS page first.

With several traits, a trait that fails does not stop the others. The CLI and One-Click Full
Analysis finish the other traits and list the failed ones in `failed_traits.txt` inside the
combined ZIP, and the CLI then exits with a code other than 0.

A run that is killed (a power cut, out of memory, a closed terminal) cannot remove its table
folder on disk, named `_trace_geno_store_<process>_<random>`. The next run in the same folder names
it in the log with its size, and you can delete it.
