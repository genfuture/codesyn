#!/bin/bash
# Quick launch script for the Web UI

cd "$(dirname "$0")"
source venv/bin/activate
python backend/main.py --ui
