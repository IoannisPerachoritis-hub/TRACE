@echo off
title TRACE
echo Starting TRACE. This takes about 15 seconds.
echo.
echo Your browser will open at http://localhost:8501
echo If the page does not load, wait a moment and refresh.
echo.
start "" http://localhost:8501
docker run --rm -p 8501:8501 ghcr.io/ioannisperachoritis-hub/trace:latest
