# run_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import re, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

_TIME_RE = re.compile(
    r"(?:(\d+)\s*d)?\s*"
    r"(?:(\d+)\s*h)?\s*"
    r"(?:(\d+)\s*m)?\s*"
    r"(?:(\d+)\s*s)?\s*$",
    re.IGNORECASE
)


def scatter3d_inputs_with_output_colorbar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    c_col: str,
    outdir: str | Path,
    filename: Optional[str] = None,
    title: Optional[str] = None,
    dpi: int = 200,
    elev: float = 20,
    azim: float = -60,
    s: float = 35,
) -> Path:
    """
    3D scatter: (x_col, y_col, z_col) as coordinates, colored by c_col.
    Saves figure to outdir and returns the saved path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cols = [x_col, y_col, z_col, c_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for 3D scatter: {missing}. Available: {list(df.columns)}")

    # Coerce to numeric and drop NaNs/infs
    work = df.copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

    if len(work) < 3:
        raise ValueError(f"Not enough valid rows for 3D scatter after cleaning: {len(work)}")

    x = work[x_col].to_numpy()
    y = work[y_col].to_numpy()
    z = work[z_col].to_numpy()
    c = work[c_col].to_numpy()

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=c, s=s)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    if title is None:
        title = f"3D scatter: ({x_col}, {y_col}, {z_col}) colored by {c_col}"
    ax.set_title(title)

    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.8)
    cbar.set_label(c_col)

    fig.tight_layout()

    if filename is None:
        filename = f"scatter3d_{x_col}_{y_col}_{z_col}_color_{c_col}.pdf"

    outpath = outdir / filename
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def one_param_change_effects(
    df: pd.DataFrame,
    outdir: str | Path,
    hyperparams=("n_layer", "n_head", "n_embd", "batch_size"),
    loss_col="best_val_loss",
    time_col_seconds="train_time_seconds",
) -> dict[str, pd.DataFrame]:
    """
    For each hyperparameter p in hyperparams:
      - hold the others fixed
      - compare all runs that share the fixed tuple
      - keep only comparisons where ONLY p differs
      - report delta loss/time for each pair

    Returns dict: {param_name -> result_df}
    Also writes CSVs into outdir.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure numeric
    work = df.copy()
    for c in list(hyperparams) + [loss_col, time_col_seconds]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # Drop rows missing essentials
    work = work.dropna(subset=[loss_col, time_col_seconds, *hyperparams]).copy()

    results: dict[str, pd.DataFrame] = {}

    for p in hyperparams:
        fixed = [q for q in hyperparams if q != p]

        rows = []
        # group by fixed params
        for fixed_vals, g in work.groupby(fixed, dropna=False):
            g = g.sort_values(p)
            if len(g) < 2:
                continue

            # Compare all pairs within the group (only p can differ because group fixed others)
            idxs = list(g.index)
            for i, j in itertools.combinations(idxs, 2):
                r1 = g.loc[i]
                r2 = g.loc[j]

                # sanity: check only p differs (should be true by construction, but keep it strict)
                if any(r1[q] != r2[q] for q in fixed):
                    continue
                if r1[p] == r2[p]:
                    continue

                delta_p = float(r2[p] - r1[p])
                delta_loss = float(r2[loss_col] - r1[loss_col])
                delta_time_s = float(r2[time_col_seconds] - r1[time_col_seconds])

                rows.append({
                    **{q: r1[q] for q in fixed},  # fixed context
                    f"{p}_from": float(r1[p]),
                    f"{p}_to": float(r2[p]),
                    f"delta_{p}": delta_p,
                    "loss_from": float(r1[loss_col]),
                    "loss_to": float(r2[loss_col]),
                    "delta_loss": delta_loss,
                    "time_s_from": float(r1[time_col_seconds]),
                    "time_s_to": float(r2[time_col_seconds]),
                    "delta_time_seconds": delta_time_s,
                    "delta_time_hours": delta_time_s / 3600.0,
                    "run_from": r1.get("run", ""),
                    "run_to": r2.get("run", ""),
                })

        res = pd.DataFrame(rows)
        if not res.empty:
            # Add a few convenience summaries
            res["loss_improved"] = res["delta_loss"] < 0
            res["time_increased"] = res["delta_time_seconds"] > 0

            # Sort: biggest loss improvements first (most negative delta_loss)
            res = res.sort_values(["delta_loss", "delta_time_seconds"], ascending=[True, True])

        results[p] = res
        res.to_csv(outdir / f"one_param_effect_{p}.csv", index=False)

    return results


def summarize_one_param_effects(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Condenses each one-param analysis into a small summary table:
      - number of valid comparisons
      - mean delta loss / time
      - fraction where loss improved
    """
    rows = []
    for p, dfp in results.items():
        if dfp is None or dfp.empty:
            rows.append({
                "param": p,
                "n_comparisons": 0,
                "mean_delta_loss": np.nan,
                "mean_delta_time_hours": np.nan,
                "frac_loss_improved": np.nan,
            })
            continue
        rows.append({
            "param": p,
            "n_comparisons": int(len(dfp)),
            "mean_delta_loss": float(dfp["delta_loss"].mean()),
            "mean_delta_time_hours": float(dfp["delta_time_hours"].mean()),
            "frac_loss_improved": float((dfp["delta_loss"] < 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("n_comparisons", ascending=False)


def parse_training_time_to_seconds(x: Any) -> float:
    """
    Accepts:
      - seconds as numeric
      - strings like "0d 17h 13m 17s" (any subset)
      - strings like "17h 13m 17s"
    Returns seconds (float). NaN if not parseable.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if s.isdigit():
        return float(s)

    m = _TIME_RE.match(s)
    if not m:
        return np.nan

    d = int(m.group(1)) if m.group(1) else 0
    h = int(m.group(2)) if m.group(2) else 0
    mm = int(m.group(3)) if m.group(3) else 0
    sec = int(m.group(4)) if m.group(4) else 0
    return float(d * 86400 + h * 3600 + mm * 60 + sec)


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def infer_hparams_from_run_name(run: str) -> Dict[str, Optional[float]]:
    """
    Parses names like:
      n_layer4_n_head4_n_embd16_batch_size64
    Returns dict with numeric values when present.
    """
    if not isinstance(run, str):
        return {"n_layer": None, "n_head": None, "n_embd": None, "batch_size": None}

    def grab(key: str) -> Optional[float]:
        m = re.search(rf"{key}(\d+)", run)
        return float(m.group(1)) if m else None

    return {
        "n_layer": grab("n_layer"),
        "n_head": grab("n_head"),
        "n_embd": grab("n_embd"),
        "batch_size": grab("batch_size"),
    }


def seconds_to_hms(seconds: float) -> str:
    if seconds is None or np.isnan(seconds):
        return "NaN"
    seconds = int(round(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}d {h:02d}h {m:02d}m {s:02d}s"


def plot_one_param_slices(
    df: pd.DataFrame,
    param: str,
    loss_col: str = "best_val_loss",
    time_col_s: str = "train_time_seconds",
    hyperparams: Tuple[str, ...] = ("n_layer", "n_head", "n_embd", "batch_size"),
    outdir: str | Path = "run_analysis_out",
    min_points_per_slice: int = 2,
    # Legend controls
    show_legend: bool = True,
    legend_max_entries: int = 12,         # limit clutter; set None to show all
    legend_outside: bool = True,          # place legend outside plot
) -> Tuple[Path, Path]:
    """
    For a chosen `param`, plot:
      1) loss vs param
      2) time vs param (hours)
    for each "slice" where all other hyperparams are fixed.

    Legend now works for any param because we don't auto-disable it when slices > 10.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if param not in hyperparams:
        raise ValueError(f"param must be in {hyperparams}, got {param}")

    fixed = [p for p in hyperparams if p != param]

    work = df.copy()
    # numeric coercion
    for c in [*hyperparams, loss_col, time_col_s]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=[param, *fixed, loss_col, time_col_s]).copy()
    if work.empty:
        raise ValueError("No valid rows after dropping NaNs for the requested plot.")

    def _as_tuple(x):
        return x if isinstance(x, tuple) else (x,)

    def _label_from_fixed(fixed_vals):
        fixed_vals = _as_tuple(fixed_vals)
        parts = []
        for k, v in zip(fixed, fixed_vals):
            # pretty int formatting
            if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
                parts.append(f"{k}={int(v)}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def _apply_legend(fig, ax):
        if not show_legend:
            return
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return

        if legend_max_entries is not None and len(handles) > legend_max_entries:
            handles = handles[:legend_max_entries]
            labels = labels[:legend_max_entries]
            labels[-1] = labels[-1] + "  (…more)"

        if legend_outside:
            ax.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
            fig.tight_layout(rect=[0, 0, 0.78, 1])  # leave room for legend
        else:
            ax.legend(handles, labels, fontsize=8, loc="best")
            fig.tight_layout()

    # ---------- Plot 1: Loss vs param ----------
    fig1 = plt.figure(figsize=(9, 5))
    ax1 = plt.gca()

    # group by fixed params
    for fixed_vals, g in work.groupby(fixed, dropna=False):
        g = g.sort_values(param)
        if len(g) < min_points_per_slice:
            continue
        x = g[param].values
        y = g[loss_col].values
        ax1.plot(x, y, marker="o", linewidth=1.2, label=_label_from_fixed(fixed_vals))

    ax1.set_xlabel(param)
    ax1.set_ylabel("Best validation loss")
    ax1.set_title(f"Effect of {param} on validation loss (others fixed)")
    ax1.grid(True)
    _apply_legend(fig1, ax1)

    p1 = outdir / f"slices_{param}_vs_loss.pdf"
    fig1.savefig(p1, dpi=200)
    plt.close(fig1)

    # ---------- Plot 2: Time vs param ----------
    fig2 = plt.figure(figsize=(9, 5))
    ax2 = plt.gca()

    for fixed_vals, g in work.groupby(fixed, dropna=False):
        g = g.sort_values(param)
        if len(g) < min_points_per_slice:
            continue
        x = g[param].values
        y = (g[time_col_s].values / 3600.0)
        ax2.plot(x, y, marker="o", linewidth=1.2, label=_label_from_fixed(fixed_vals))

    ax2.set_xlabel(param)
    ax2.set_ylabel("Training time [hours]")
    ax2.set_title(f"Effect of {param} on training time (others fixed)")
    ax2.grid(True)
    _apply_legend(fig2, ax2)

    p2 = outdir / f"slices_{param}_vs_time.pdf"
    fig2.savefig(p2, dpi=200)
    plt.close(fig2)

    return p1, p2


# ----------------------------
# Main analysis
# ----------------------------

@dataclass
class AnalysisResult:
    best_run: str
    best_val_loss: float
    best_time_seconds: float
    correlations: pd.DataFrame
    cleaned_table: pd.DataFrame
    saved_plots: List[Path]


def analyze_runs_csv(
    csv_path: str | Path,
    base: str | Path = "runs/sweeps",
    outdir: str | Path = "run_analysis_out",
    run_col: str = "run",
    val_loss_col: str = "best_val_loss",
    time_col: str = "train_time",
    time_is_seconds: bool = False,
) -> AnalysisResult:
    """
    Reads a CSV of runs and produces:
      - bar plot of best val loss per run
      - bar plot of train time per run (hours)
      - scatter plot: time vs loss
      - scatter plots: each hyperparam vs loss/time
      - correlation matrix among numeric hyperparams + loss + time

    Required columns:
      - run_col (default: 'run')
      - val_loss_col (default: 'best_val_loss')
      - time_col (default: 'train_time') : either seconds or strings like '0d 17h 13m 17s'

    If your CSV has hyperparams as columns already, it will use them too.
    If not, it will try to infer n_layer/n_head/n_embd/batch_size from the run name.
    """
    base = Path(base)  
    csv_path = base / csv_path
    #outdir = csv_path / outdir
    outdir = base / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    if run_col not in df.columns:
        raise ValueError(f"Missing required column '{run_col}'. Columns: {list(df.columns)}")
    if val_loss_col not in df.columns:
        raise ValueError(f"Missing required column '{val_loss_col}'. Columns: {list(df.columns)}")
    if time_col not in df.columns:
        raise ValueError(f"Missing required column '{time_col}'. Columns: {list(df.columns)}")

    # Normalize core columns
    df = df.copy()
    df[val_loss_col] = pd.to_numeric(df[val_loss_col], errors="coerce")

    if time_is_seconds:
        df["_time_seconds"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_seconds"] = df[time_col].apply(parse_training_time_to_seconds)

    # Add inferred hyperparams if missing
    for hp in ["n_layer", "n_head", "n_embd", "batch_size"]:
        if hp not in df.columns:
            df[hp] = df[run_col].apply(lambda r: infer_hparams_from_run_name(str(r)).get(hp))

    df = ensure_numeric(df, ["n_layer", "n_head", "n_embd", "batch_size", "_time_seconds", val_loss_col])

    # Drop rows without essential metrics
    core = df.dropna(subset=[val_loss_col, "_time_seconds"]).copy()
    if core.empty:
        raise ValueError("After parsing, no rows have both val loss and training time available.")

    core["_time_hours"] = core["_time_seconds"] / 3600.0

    # Best run by val loss
    best_idx = core[val_loss_col].idxmin()
    best_row = core.loc[best_idx]
    best_run = str(best_row[run_col])
    best_val_loss = float(best_row[val_loss_col])
    best_time_seconds = float(best_row["_time_seconds"])

    saved: List[Path] = []

    # Sort for plotting
    core_sorted_loss = core.sort_values(val_loss_col, ascending=True)
    core_sorted_time = core.sort_values("_time_seconds", ascending=True)

    # Plot 1: Best val loss per run (bar)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(core_sorted_loss[run_col].astype(str), core_sorted_loss[val_loss_col].values)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.ylabel("Best validation loss")
    plt.title("Best validation loss per run (lower is better)")
    plt.tight_layout()
    p = outdir / "bar_best_val_loss.pdf"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    saved.append(p)

    # Plot 2: Training time per run (bar, hours)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(core_sorted_time[run_col].astype(str), core_sorted_time["_time_hours"].values)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.ylabel("Training time [hours]")
    plt.title("Training time per run")
    plt.tight_layout()
    p = outdir / "bar_training_time_hours.pdf"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    saved.append(p)

    # Plot 3: Scatter time vs loss
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(core["_time_hours"].values, core[val_loss_col].values)
    plt.xlabel("Training time [hours]")
    plt.ylabel("Best validation loss")
    plt.title("Time vs validation loss")
    plt.tight_layout()
    p = outdir / "scatter_time_vs_loss.pdf"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    saved.append(p)

    # Hyperparam vs loss/time scatters (only if enough non-NaN)
    for hp in ["n_layer", "n_head", "n_embd", "batch_size"]:
        if hp in core.columns and core[hp].notna().sum() >= 2:
            # hp vs loss
            fig = plt.figure(figsize=(6, 5))
            plt.scatter(core[hp].values, core[val_loss_col].values)
            plt.xlabel(hp)
            plt.ylabel("Best validation loss")
            plt.title(f"{hp} vs validation loss")
            plt.tight_layout()
            p = outdir / f"scatter_{hp}_vs_loss.pdf"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            saved.append(p)

            # hp vs time
            fig = plt.figure(figsize=(6, 5))
            plt.scatter(core[hp].values, core["_time_hours"].values)
            plt.xlabel(hp)
            plt.ylabel("Training time [hours]")
            plt.title(f"{hp} vs training time")
            plt.tight_layout()
            p = outdir / f"scatter_{hp}_vs_time.pdf"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            saved.append(p)

    # Correlations (Pearson) on numeric columns
    numeric_cols = [c for c in ["n_layer", "n_head", "n_embd", "batch_size"] if c in core.columns]
    corr_df = core[numeric_cols + [val_loss_col, "_time_seconds"]].corr(numeric_only=True)

    # Save correlations as CSV
    corr_path = outdir / "correlations.csv"
    corr_df.to_csv(corr_path)
    saved.append(corr_path)

    # Save cleaned table for inspection
    cleaned = core[[run_col, val_loss_col, "_time_seconds", "_time_hours"] + numeric_cols].copy()
    cleaned = cleaned.sort_values(val_loss_col, ascending=True)
    cleaned_path = outdir / "cleaned_runs_table.csv"
    cleaned.to_csv(cleaned_path, index=False)
    saved.append(cleaned_path)

    # Example integration inside analyze_runs_csv(...) after 'cleaned' is built:
    effects = one_param_change_effects(
        df=cleaned.rename(columns={
            "_time_seconds": "train_time_seconds",
            # ensure your loss column name matches below
        }),
        outdir=outdir,
        hyperparams=("n_layer", "n_head", "n_embd", "batch_size"),
        loss_col=val_loss_col,
        time_col_seconds="train_time_seconds",
    )

    summary = summarize_one_param_effects(effects)
    summary.to_csv(Path(outdir) / "one_param_effects_summary.csv", index=False)
    print("\n=== One-param change summary ===")
    print(summary)

    # Example: layers effect with heads/embd/batch fixed
    for p in ["n_layer", "n_head", "n_embd", "batch_size"]:
        plot_one_param_slices(
            df=cleaned.rename(columns={
                "_time_seconds": "train_time_seconds",
                # ensure your loss column name matches below
            }),
            param=p,
            loss_col="best_val_loss",
            time_col_s="train_time_seconds",
            outdir=outdir,
            min_points_per_slice=2,  # set to 3 if you literally want “only the 3-point cases”
        )


    # 3D scatter: pick any 3 inputs + 1 output to color
    # Example: inputs = (n_layer, n_head, n_embd), output = best_val_loss
    p3d = scatter3d_inputs_with_output_colorbar(
        df=cleaned,
        x_col="n_layer",
        y_col="n_head",
        z_col="n_embd",
        c_col=val_loss_col,   # or "_time_hours" if you want time as the "output"
        outdir=outdir,
        filename="scatter3d_layers_heads_embd_color_loss.pdf",
        title="Hyperparams (3D) colored by validation loss",
    )
    saved.append(p3d)

    # Optional second one: color by training time
    p3d_time = scatter3d_inputs_with_output_colorbar(
        df=cleaned,
        x_col="n_layer",
        y_col="n_head",
        z_col="n_embd",
        c_col="_time_hours",
        outdir=outdir,
        filename="scatter3d_layers_heads_embd_color_time.pdf",
        title="Hyperparams (3D) colored by training time [hours]",
    )
    saved.append(p3d_time)



    return AnalysisResult(
        best_run=best_run,
        best_val_loss=best_val_loss,
        best_time_seconds=best_time_seconds,
        correlations=corr_df,
        cleaned_table=cleaned,
        saved_plots=saved,
    )


"""
python run_analysis.py --csv <name_sweep>.csv
"""

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to runs CSV")
    ap.add_argument("--base", default="runs/sweeps", help="Base directory for outputs")
    ap.add_argument("--outdir", default="sweep_analysis_out", help="Output directory for plots/tables")
    ap.add_argument("--run_col", default="run")
    ap.add_argument("--val_loss_col", default="best_val_loss")
    ap.add_argument("--time_col", default="train_time")
    ap.add_argument("--time_is_seconds", action="store_true", help="If set, time_col is already in seconds")
    args = ap.parse_args()

    res = analyze_runs_csv(
        csv_path=args.csv,
        outdir=args.outdir,
        base=args.base,
        run_col=args.run_col,
        val_loss_col=args.val_loss_col,
        time_col=args.time_col,
        time_is_seconds=args.time_is_seconds,
    )

    print("\n================= BEST RUN =================")
    print(f"Run: {res.best_run}")
    print(f"Best validation loss: {res.best_val_loss:.12g}")
    print(f"Training time: {seconds_to_hms(res.best_time_seconds)}")
    print("\n============= CORRELATIONS (Pearson) =============")
    print(res.correlations.round(4))
    print("\nSaved outputs:")
    for p in res.saved_plots:
        print(f" - {p}")
