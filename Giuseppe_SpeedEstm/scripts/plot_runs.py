import argparse, os, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()
    plt.figure()
    for rd in args.run_dirs:
        p = os.path.join(rd, "history.csv")
        if not os.path.exists(p):
            print(f"Missing {p}")
            continue
        df = pd.read_csv(p)
        if "train_total" in df:
            plt.plot(df["epoch"], df["train_total"], label=os.path.basename(rd))
    plt.xlabel("epoch"); plt.ylabel("train_total"); plt.legend(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
