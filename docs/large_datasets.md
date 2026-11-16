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

`run_metadata.json` in the results ZIP records the mode (`Genotype table`), how it was chosen
(`Genotype table choice`: `auto` or the option) and why (`Genotype table reason`, for example
"auto: the in-memory estimate did not fit in the free memory"), with the estimate and the number
of genotype calls of the genotypes as read (`Memory estimate, in-memory run (GB)` and
`Genotype calls`). The free memory changes from run to run, so it is in the log line only: two
runs of the same analysis give identical ZIPs. (The log line's estimate is made before reading,
from the file size, and can be higher than the recorded one.) The first three keys are the only
difference between the ZIPs of the two modes; the HTML report is the same.

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
run records the libraries and their versions (numpy and scipy each bring their own, so both are
named, numpy's first), the thread count, the cores, `--boot-jobs` and the threads per worker in
`run_metadata.json` and in the report's provenance table.
To reproduce a run exactly, set the same thread count, for example `OPENBLAS_NUM_THREADS=4`.

## Docker

The default folder for the table file is inside `--output`, so the README's bind mount (and
`--user "$(id -u):$(id -g)"` on Linux) gives TRACE a folder it can write. A Docker named volume
works if the user TRACE runs as can write to it; a new volume is owned by root, so TRACE's image
user cannot write there unless you give the volume to that user. A container memory limit is read
by `auto`: a run that does not fit under the limit keeps its table on disk.

## In the GUI

**Opening a big VCF.** Next to the upload box, the GWAS page has a box "Or type the VCF's location
on this computer". It takes a `.vcf` or `.vcf.gz` file on the computer that runs TRACE, checks
that it exists and can be read, and shows its size. Nothing is uploaded or copied: TRACE reads the
file where it is. The upload box takes files up to 2 GB; an upload is held in the server's memory
while it arrives (twice at its end), before TRACE can check anything, so bigger files should go
through the location box. An uploaded file is written once to the work folder (below) and read
from there; TRACE then removes the server's copy of the upload from memory and empties the upload
box. A copy that fails (a full disk, for one) leaves nothing in the work folder, and the page says
why. It is not tried again on every rerun, only when you upload the file again or choose another
work folder.

A location that fails the check (a typo, another file type) stops the page with the reason; the
file an earlier location gave is not used under the new text.

TRACE knows a VCF by its content, hashed once for each location, size and modification time. A
file rewritten in place with the same size and the same modification time (a copy that keeps the
old file's time stamp, for one) is therefore read as the old one until Clear session data or a
restart of the server.

On a shared server, whose users should not open the server's own files, set
`TRACE_DISABLE_PATH_INPUT=1` to hide the location box. In Docker, the location is a path inside
the container: the folder you mounted (for example `/data/panel.vcf.gz` for `-v
"$PWD:/data"`).

**Where the table is kept.** A collapsed section "Large datasets" on the GWAS page offers
Automatic (the default), In memory or On disk, with the same meaning as `--geno-store`. Before
anything is read, the page shows the same check the CLI logs (how many genotype calls, the memory
an in-memory run needs, the free memory) and stops with a plain message when the choice cannot
work. The environment variable `TRACE_GENO_STORE` (`auto`, `memory` or `disk`) sets the default,
so a server or a Docker image can force it.

**The GUI's memory estimate.** An open GUI session holds more than a CLI run: Streamlit itself,
the page's figures and downloads, the session's own tables, the kinship and PCs of the last
settings, and, for each SVG download, a headless Chrome that kaleido starts. So the page's check
and One-Click's add to the CLI's estimate 600 MiB, 8 bytes per genotype call, 512 bytes per pair
of samples, and 750 MiB plus 2.4 KiB per SNP (up to 500,000 SNPs) for Chrome. The basis is the
GUI's measured peak (the whole process tree, with every cache full, three models, subsampling
and SVG exports), which lies at least 14% below this estimate: 1,493 MiB against 2,554 MiB for
the first example, 2,633 against 3,078 for tomato (166 samples x 44,317 SNPs), and 4,445 against
6,043 for 1,000 samples x 50,000 SNPs. Automatic chooses by this estimate and the page shows it;
`run_metadata.json` records the CLI's, so the metadata is the same whichever program ran.
Chrome is always counted, which is cautious on a server without Chrome (the page then offers
HTML downloads only).

**Work folder.** Saved uploads and tables kept on disk go into the work folder: the environment
variable `TRACE_GENO_STORE_DIR`, else a folder `trace_gui` in the system's temporary folder. The
same section can change it, and the check includes its free space. Each table file gets its own
folder, `_trace_geno_store_gui_<process>_<random>`, and each saved upload its own
`_trace_upload_gui_<process>_<random>`; TRACE keeps at most two tables, and a third while a new
one is built, and removes a folder when its table is no longer kept and when the server stops
normally. Table folders and saved uploads left by a run or a server that did not stop are named
on the page with their size; they are never deleted automatically.

**What the pages do with the table on disk.** The GWAS page runs its preprocessing (sample
matching, phenotype and SNP QC, the fill of missing calls, kinship, the QC report) on the table
at the same places as the CLI. MLM, MLMM, FarmCPU, the OLS effects, LD decay, M_eff and
subsampling read it in pieces. One-Click uses the same table, and its `run_metadata.json` records
where the table was kept, as the CLI's does. The Post-GWAS page reads only the window or block
each tab shows. The results are the same as with the table in memory, byte for byte. Not
available with the table on disk: LD-kNNi imputation (choose mean imputation, or keep the table
in memory), and the haplotype PCA panel in a session without the GWAS page's PCs, since the panel
would then compute its own from the whole table (it says why). Every GWAS page run leaves those
PCs (its PCA, up to 20 PCs, whatever number of PCs the models use).

**Subsampling.** With the table on disk, the iterations run one after another. In memory with
more than one core, the page first checks that the workers fit in the free memory (each worker
after the first needs about 300 MiB and 12 bytes per genotype call). If they do not, it runs the
iterations one after another and says so. The results are the same either way. When the
iterations run one after another with more than one core chosen, each uses the thread count a
worker would have, and that limit holds for the whole server process while it runs (one
subsampling at a time sets it). On a shared server, other sessions' computations that run
meanwhile may then use fewer threads, and their last digits differ, as between thread counts
(see "Reproducibility" above).

**Tables shared by sessions.** A table belongs to the server, not to one browser tab: sessions
that open the same VCF with the same choice share one table, and the server keeps at most two,
and a third while a new one is built. It builds one table at a time: a session that needs a new
table while another is built waits for that build (its page says "Waiting for another session's
genotype table to be built…" meanwhile), and then checks the free memory and disk again, so the
check counts what that build used. The page stops with a plain message when they
no longer suffice, by the rule of the choice: for Automatic, when an in-memory run would now
need more than 80% of the free memory (the next run then keeps the table on disk); for In
memory, more than all of it. When a third is needed, the one used least recently is closed once
the new one is open (a file that cannot be read closes nothing), before the next build starts.
A session that still held a closed table (closed this way, or by Clear session data in another
session) reads the VCF again: the GWAS page on its next run, One-Click for the trait it was
running and the ones after it, each with a note on the page. A Run GWAS that was reading the
table at that moment stops with a message, and the next run reads it again. Clear session data
in another session also deletes the uploads the server saved, so a VCF that was uploaded cannot
be read again: the page then says that the uploaded VCF is no longer in the work folder and asks
you to upload it again, on its next run or when it would read the VCF again. The same happens
when the system cleans its temporary folder, where the work folder is by default; on a server
that runs for days, set `TRACE_GENO_STORE_DIR` to a folder the system does not clean. A VCF
given by its location is read again. The Post-GWAS page asks you to run the GWAS page again. A
table kept in memory is not closed: it stays in memory while a session still holds it, until
that session's next run.

**Clear session data** frees this server's genotype tables (in memory, and the table files with
their folders on disk), the uploads it saved, every cached result, and the cache files that
earlier versions of TRACE left in Streamlit's cache folder (`~/.streamlit/cache`; other apps' files
there are not touched). It also drops this session's data and results on both pages, and
empties the VCF location box, the upload boxes and the trait selection, so nothing is read again
until you give the files again; the other settings stay. On a shared server Clear affects every
session, since the tables and the saved uploads are the server's: another session's uploaded VCF
must then be uploaded again.

**The interactive Manhattan.** Above 500,000 SNPs it draws every SNP with p ≤ 0.01 and one in k of
the others, in genome order, with k the smallest whole number that keeps the drawing at 500,000
points or fewer. A caption under the figure says so ("N of M SNPs drawn: ..."). The static
Manhattan and the results table keep every SNP. The reason is the browser: Streamlit sends a
figure in one message, of at most 200 MB by default, and at 1,070,795 SNPs the full figure was
157 MB; the thinned one is 53 MB, and a figure at the limit 73 MB. Below the limit nothing
changes. The cross-model "Overlay Manhattan by model" is thinned when the points of all its
models together are more than 500,000 (three models of 200,000 SNPs are 600,000 points): within
each model it draws every SNP with p ≤ 0.01 and one in k of the others, with the same k for every
model, so every region shows each model's points alike. Its caption counts points ("N of M points
drawn (S SNPs x 3 models): ...").

## Checking that both modes agree

Run the same analysis twice, once with each mode, into two folders, and compare the two ZIPs:

```bash
trace-gwas --vcf my.vcf.gz --pheno pheno.csv --trait Yield --output run_memory/ --geno-store memory
trace-gwas --vcf my.vcf.gz --pheno pheno.csv --trait Yield --output run_disk/ --geno-store disk
python scripts/compare_result_zips.py run_memory/GWAS_*.zip run_disk/GWAS_*.zip
```

The script compares every file of the two ZIPs byte for byte, leaving out only the
`run_metadata.json` keys above and the report's "Generated" time.
