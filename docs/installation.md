# Installation

TRACE runs on Linux, macOS and Windows. There are three routes. Pick the one whose requirement
you already meet: Docker, [uv](https://docs.astral.sh/uv/), or Python 3.11 to 3.13 with pip.

## Docker (nothing else to install)

```bash
docker run -p 8501:8501 ghcr.io/ioannisperachoritis-hub/trace:latest
```

Open <http://localhost:8501>, upload a VCF and a phenotype table, and download the results ZIP
before you close the container. Nothing is kept between sessions, unless you give the container a
folder (see [Troubleshooting](troubleshooting.md#docker-trace-cannot-write-to-a-named-volume)).

Once pulled, `docker run` uses the local copy and never looks for a newer release, so update it
yourself:

```bash
docker pull ghcr.io/ioannisperachoritis-hub/trace:latest
```

On a Mac with an Apple chip, add `--platform linux/amd64` to `docker pull` and `docker run`. The
image is built for Intel and AMD processors only, so Docker runs it under emulation, more slowly.

For the CLI in the container, mount a working folder at `/data` and run the container as your own
user, so that the container can write there and the results belong to you:

```bash
mkdir -p data/results
docker run --rm --user "$(id -u):$(id -g)" --entrypoint trace-gwas \
    -v "$(pwd)/data:/data" \
    ghcr.io/ioannisperachoritis-hub/trace:latest \
    --vcf /data/my_genotypes.vcf.gz --pheno /data/my_pheno.csv \
    --trait MyTrait --output /data/results/
```

On Docker Desktop for Windows, `--user` is not needed: the container can write to a mounted
Windows folder, and the results belong to your Windows account. PowerShell has no `id` and no `\`
line continuation, so run the command on one line:

```powershell
docker run --rm -v "${PWD}\data:/data" --entrypoint trace-gwas ghcr.io/ioannisperachoritis-hub/trace:latest --vcf /data/my_genotypes.vcf.gz --pheno /data/my_pheno.csv --trait MyTrait --output /data/results/
```

Mount your data at `/data`, not at `/app`: `/app` holds TRACE itself.

To build the image yourself:

```bash
docker build -t trace .
docker run -p 8501:8501 trace
```

## uv (no administrator rights needed)

uv is a single program that installs into your home folder and fetches Python itself, so this
route works on a managed computer where Docker cannot be installed.

```bash
git clone https://github.com/IoannisPerachoritis-hub/TRACE.git && cd TRACE
uv venv --python 3.11
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -e .
streamlit run app.py
```

## pip (Python 3.11 to 3.13)

TRACE is tested on Python 3.11 and 3.12.
It also installs and runs on 3.13. It does not install on 3.14 yet, because two of its libraries
(pyarrow and scikit-allel) have no ready-made packages for it.

```bash
git clone https://github.com/IoannisPerachoritis-hub/TRACE.git
cd TRACE
pip install -e .
```

This installs TRACE and its dependencies, with the `trace-gwas` command. Check it, then start the
GUI:

```bash
python -c "import streamlit, fastlmm.association; print('TRACE dependencies OK')"
streamlit run app.py
```

`requirements.txt` holds the full list of dependencies as lowest versions. For an install that
matches the versions behind the paper's benchmarks, add the exact versions CI uses (Python 3.11
or 3.12):

```bash
pip install -e . -c constraints.txt
```

## Main dependencies

| Package | Used for |
|---|---|
| streamlit | the GUI |
| fastlmm | the mixed linear model |
| pandas, numpy, scipy | tables and computation |
| statsmodels | OLS statistics |
| matplotlib, plotly | static and interactive plots |
| scikit-allel | reading VCF files |
