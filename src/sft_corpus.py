import json, os, argparse

def build_jsonl(in_jsonl: str, out_jsonl: str, add_final=False):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(in_jsonl,"r") as f, open(out_jsonl,"w") as g:
        for line in f:
            ex = json.loads(line)
            rationale_text = ex["gen"]
            final_line = f"\n#### {ex['gold']}" if add_final else ""
            prompt = f"Question: {ex['question']}\nReasoning:"
            completion = f"{rationale_text}{final_line}"
            g.write(json.dumps({"prompt": prompt, "completion": completion})+"\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--add_final", action="store_true")
    args = ap.parse_args()
    build_jsonl(args.in_jsonl, args.out_jsonl, args.add_final)
