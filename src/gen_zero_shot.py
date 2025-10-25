import os, argparse, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .data import load_gsm8k, set_seed
from .hf_auth import resolve_hf_token
from .utils import parse_final_answer
from transformers.utils import logging as hf_logging

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct"))
    ap.add_argument("--adapter", type=str, default=None, help="Optional PEFT/LoRA adapter directory")
    ap.add_argument("--split", type=str, default="test", choices=["train","test","validation","test_simplified"])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out", type=str, default="outputs/zero_shot_preds.jsonl")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    args = ap.parse_args()

    set_seed(args.seed)
    hf_logging.set_verbosity_error()

    tok = resolve_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=tok)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="auto", token=tok
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    ds = load_gsm8k(args.split, args.max_samples)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as fout:
        for ex in tqdm(ds, total=len(ds)):
            q = ex["question"].strip()
            gold = ex["answer"].split("####")[-1].strip()

            prompt = f"Q: {q}\nA: Let's think step by step, then give the final answer on a new line as #### <number>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temp > 0.0),
                temperature=args.temp if args.temp > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = parse_final_answer(text) or ""

            fout.write(json.dumps({"question": q, "gold": gold, "gen": text, "pred": pred}) + "\n")

if __name__ == "__main__":
    main()
