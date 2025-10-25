import os, argparse, json
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, DataCollatorForLanguageModeling,
                          Trainer)
from peft import LoraConfig, get_peft_model, TaskType
from .hf_auth import resolve_hf_token

def load_jsonl_as_hf(jsonl_path):
    return load_dataset("json", data_files=jsonl_path)["train"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.getenv("MODEL_ID"))
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--out_dir", default="outputs/sft")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    tok = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=tok)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, token=tok, device_map="auto", load_in_4bit=args.qlora
    )

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM, target_modules=["q_proj","k_proj","v_proj","o_proj"]
    )
    model = get_peft_model(model, peft_cfg)

    ds = load_jsonl_as_hf(args.data_jsonl)

    def tok_fmt(ex):
        text = ex["prompt"] + "\n" + ex["completion"]
        return tokenizer(text, truncation=True, max_length=1024)
    tds = ds.map(tok_fmt, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    os.makedirs(args.out_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.out_dir, per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, logging_steps=50, save_strategy="no",
        fp16=args.fp16
    )
    trainer = Trainer(model=model, args=targs, train_dataset=tds, data_collator=collator)
    trainer.train()
    model.save_pretrained(os.path.join(args.out_dir, "lora"))

if __name__ == "__main__":
    main()
