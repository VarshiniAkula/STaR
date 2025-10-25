#!/usr/bin/env bash
set -e
source .venv/bin/activate

# (A) First, generate rationales from base model on TRAIN split
python -m src.gen_star --split train --out_prefix outputs/sft_from_base --with_rationalization

# (B) Build SFT corpus (rationales only)
python -m src.sft_corpus --in_jsonl outputs/sft_from_base.gen_correct.jsonl \
       --out_jsonl outputs/sft_train.jsonl

# (C) Train SFT
python -m src.train_sft --data_jsonl outputs/sft_train.jsonl --out_dir outputs/sft_vanilla \
       --qlora --fp16

# (D) Evaluate the SFT model zero-shot-CoT on TEST
python -m src.gen_zero_shot --model_id outputs/sft_vanilla/lora \
       --split test --out outputs/sft_zero_shot_test.jsonl
python -m src.eval_exact_match --preds outputs/sft_zero_shot_test.jsonl
