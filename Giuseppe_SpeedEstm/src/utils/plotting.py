import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_history(history: list[dict], outdir: str):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(outdir, "history.csv"), index=False)
    plt.figure()
    if "train_total" in df:
        plt.plot(df["epoch"], df["train_total"], label="train_total")
    for c in df.columns:
        if c.startswith("val_"):
            plt.plot(df["epoch"], df[c], label=c)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves.png"))
    plt.close()
