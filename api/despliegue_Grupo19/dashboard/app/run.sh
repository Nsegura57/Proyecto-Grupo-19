#!/bin/bash
export PORT=${PORT:-8050}
gunicorn app:server -b :$PORT --workers 2 --timeout 120
