# One-click launchers

Double-click to start TRACE in your browser without opening a terminal. Each
script runs the published Docker image and opens <http://localhost:8501>.

| Platform | Script |
|----------|--------|
| Windows  | `TRACE.bat` |
| macOS    | `TRACE.command` |
| Linux    | `trace.sh` |

**Requires [Docker](https://docs.docker.com/get-started/get-docker/)** to be
installed and running. The first launch pulls the image (~1.9 GB, a few
minutes); later launches reuse the local copy. To update, pull explicitly:

    docker pull ghcr.io/ioannisperachoritis-hub/trace:latest

## Known limitations

- **The browser may open a second or two before the app finishes booting** — if
  the page does not load, wait a moment and refresh.
- **First-run security prompts on a downloaded script:** Windows shows an
  "unknown publisher" warning and macOS Gatekeeper blocks it. On macOS,
  right-click the script and choose **Open** to run it the first time.
