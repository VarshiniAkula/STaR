#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install -r requirements.txt
echo "✅ venv ready. Remember: source .venv/bin/activate"
