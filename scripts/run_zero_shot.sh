#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.gen_zero_shot --split test --out outputs/zero_shot_test.jsonl
python -m src.eval_exact_match --preds outputs/zero_shot_test.jsonl
