import json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()
    n=0; c=0
    with open(args.preds,"r") as f:
        for line in f:
            n+=1
            ex = json.loads(line)
            c += (ex["pred"].strip()==ex["gold"].strip())
    acc = 100.0*c/max(n,1)
    print(f"Exact Match: {acc:.2f}%  ({c}/{n})")

if __name__ == "__main__":
    main()
