import os, argparse, subprocess, sys

def sh(*args):
    print("+", " ".join(args))
    return subprocess.run(args, check=True)

def line_count(path):
    cmd = f"test -f {path} && wc -l {path} | awk '{{print $1}}' || echo 0"
    out = subprocess.check_output(["bash","-lc", cmd]).decode().strip()
    return int(out)

def build_sft_jsonl(iter_prefix):
    keep = f"{iter_prefix}.gen_correct.jsonl"
    rat  = f"{iter_prefix}.rat_correct.jsonl"
    merged = f"{iter_prefix}.merged.jsonl"
    cmd = "cat"
    files = []
    if os.path.exists(keep): files.append(keep)
    if os.path.exists(rat):  files.append(rat)
    if not files:
        print(f"ERROR: no files to merge for {iter_prefix}")
        sys.exit(1)
    subprocess.run(["bash","-lc", f"{cmd} {' '.join(files)} > {merged}"], check=True)

    if line_count(merged) == 0:
        print(f"ERROR: merged JSONL is empty for {iter_prefix}")
        sys.exit(1)

    sh(sys.executable, "-m", "src.sft_corpus",
       "--in_jsonl", merged,
       "--out_jsonl", f"{iter_prefix}.sft.jsonl")
    return f"{iter_prefix}.sft.jsonl"

def run_gen(split, model_id, adapter, out_prefix, rationalization):
    cmd = [sys.executable, "-m", "src.gen_star",
           "--model_id", model_id,
           "--split", split,
           "--out_prefix", out_prefix]
    if adapter:
        cmd += ["--adapter", adapter]
    if rationalization:
        cmd += ["--with_rationalization"]  # boolean flag, no value
    sh(*cmd)

def train_sft(data_jsonl, out_dir, model_id):
    sh(sys.executable, "-m", "src.train_sft",
       "--model_id", model_id,
       "--data_jsonl", data_jsonl,
       "--out_dir", out_dir,
       "--qlora", "--fp16")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.getenv("MODEL_ID","meta-llama/Llama-3.2-3B-Instruct"))
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--rationalization", action="store_true")
    args = ap.parse_args()

    base = args.model_id

    # Iteration 0: base model -> generate train
    run_gen("train", base, None, "outputs/star_iter0", args.rationalization)
    sft0 = build_sft_jsonl("outputs/star_iter0")
    train_sft(sft0, "outputs/sft_iter0", base)

    # Iteration 1: base + adapter -> generate train
    run_gen("train", base, "outputs/sft_iter0/lora", "outputs/star_iter1", args.rationalization)
    sft1 = build_sft_jsonl("outputs/star_iter1")
    train_sft(sft1, "outputs/sft_iter1", base)

if __name__ == "__main__":
    main()
