import os, json, torch, argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.utils import logging as hf_logging

# Robust GSM8K loader (no surprises)
from datasets import load_dataset
def load_gsm8k(split, max_samples=None):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    return ds

def parse_final_answer(txt):
    # very simple numeric scrape: look for '#### <number>' at the end
    import re
    m = re.search(r"####\s*([\-+]?\d+(?:\.\d+)?)\s*$", txt)
    return m.group(1) if m else None

def build_zero_shot_prompt(q):
    return f"Q: {q}\nA: Let's think step by step, then give the final answer on a new line as #### <number>\n"

def build_hint_prompt(q, gold):
    return (f"Q: {q}\nThe correct answer is #### {gold}.\n"
            f"A: Let's reason step by step, showing why that answer is correct, "
            f"then end with #### {gold}\n")

def generate_one(model, tokenizer, prompt, max_new_tokens=512, temp=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temp > 0.0),
            temperature=temp if temp > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=os.getenv("MODEL_ID","meta-llama/Llama-3.2-3B-Instruct"))
    ap.add_argument("--adapter", type=str, default=None, help="Optional PEFT/LoRA adapter dir")
    ap.add_argument("--split", type=str, default="train", choices=["train","test"])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out_prefix", type=str, default="outputs/star_iter0")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--with_rationalization", action="store_true")
    args = ap.parse_args()

    hf_logging.set_verbosity_error()

    tok = os.getenv("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=tok)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="auto", token=tok
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    ds = load_gsm8k(args.split, args.max_samples)
    n = len(ds)
    print(f"Loaded GSM8K/{args.split}: {n} examples")
    if n == 0:
        print("ERROR: dataset is empty; aborting.")
        return

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    keep_path = f"{args.out_prefix}.gen_correct.jsonl"
    rat_path  = f"{args.out_prefix}.rat_correct.jsonl"
    with open(keep_path, "w", encoding="utf-8") as fk, open(rat_path,"w",encoding="utf-8") as fr:
        for ex in tqdm(ds, total=n, desc=f"Generating {args.split} split"):
            q = ex["question"].strip()
            gold = ex["answer"].split("####")[-1].strip()
            g = generate_one(model, tokenizer, build_zero_shot_prompt(q), args.max_new_tokens, args.temp)
            yhat = parse_final_answer(g)
            if yhat == gold:
                fk.write(json.dumps({"question": q, "gold": gold, "gen": g}) + "\n")
                continue
            if args.with_rationalization:
                h = generate_one(model, tokenizer, build_hint_prompt(q, gold), args.max_new_tokens, args.temp)
                yhat2 = parse_final_answer(h)
                if yhat2 == gold:
                    fr.write(json.dumps({"question": q, "gold": gold, "gen": h}) + "\n")

if __name__ == "__main__":
    main()
