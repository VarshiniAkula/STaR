#!/usr/bin/env bash
set -e
source .venv/bin/activate

# Outer loop (configurable)
python -m src.star_loop --iterations 2 --rationalization

# Final evaluation: last SFT checkpoint
python -m src.gen_zero_shot --model_id outputs/sft_iter1/lora \
       --split test --out outputs/star_final_test.jsonl
python -m src.eval_exact_match --preds outputs/star_final_test.jsonl
