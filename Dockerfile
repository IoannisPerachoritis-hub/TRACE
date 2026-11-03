FROM python:3.11-slim@sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser \
    && chown appuser:appuser /app

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY --chown=appuser:appuser . .

RUN pip install --no-cache-dir . -c constraints.txt

USER appuser

# Under --user "$(id -u):$(id -g)" (README, Getting started) the uid usually has no
# passwd entry in the image, so Docker sets HOME=/ and matplotlib cannot create its
# config dir ("mkdir -p failed for path /.config/matplotlib"). Point both at /tmp,
# which every uid can write, so no library falls back with a warning.
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
