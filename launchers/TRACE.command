#!/bin/bash
echo "Starting TRACE..."
sleep 2
open http://localhost:8501
docker run --rm -p 8501:8501 ghcr.io/ioannisperachoritis-hub/trace:latest
