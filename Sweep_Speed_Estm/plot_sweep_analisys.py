# run_analysis.py       | SET (leave it as it is)
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


def pick_best_tradeoff_run(
    df: pd.DataFrame,
    run_col: str = "run",
    loss_col: str = "best_val_loss",
    time_col_s: str = "_time_seconds",
    w_loss: float = 0.7,
    w_time: float = 0.3,
    scaling: str = "minmax",   # "minmax" | "robust"
    outdir: str | Path = "run_analysis_out",
    filename_prefix: str = "tradeoff",
    dpi: int = 200,
    ax_pareto: Optional[plt.Axes] = None,
    ax_score: Optional[plt.Axes] = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Same scoring, but can draw into provided axes.
    If axes are not provided, it creates its own figure(s) and saves PDFs.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if abs((w_loss + w_time) - 1.0) > 1e-9:
        raise ValueError("Require w_loss + w_time == 1.0")

    work = df.copy()
    for c in [loss_col, time_col_s]:
        if c not in work.columns:
            raise ValueError(f"Missing column '{c}'.")
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[loss_col, time_col_s]).copy()
    if work.empty:
        raise ValueError("No valid rows after cleaning.")

    loss = work[loss_col].to_numpy(dtype=float)
    time_s = work[time_col_s].to_numpy(dtype=float)

    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _robust(x: np.ndarray) -> np.ndarray:
        med = np.nanmedian(x)
        q1 = np.nanpercentile(x, 25)
        q3 = np.nanpercentile(x, 75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr < 1e-12:
            return np.zeros_like(x)
        z = (x - med) / iqr
        return 1.0 / (1.0 + np.exp(-z))

    if scaling == "minmax":
        loss_s = _minmax(loss)
        time_s_scaled = _minmax(time_s)
    elif scaling == "robust":
        loss_s = _robust(loss)
        time_s_scaled = _robust(time_s)
    else:
        raise ValueError("scaling must be 'minmax' or 'robust'")

    work["_loss_scaled"] = loss_s
    work["_time_scaled"] = time_s_scaled
    work["_tradeoff_score"] = w_loss * loss_s + w_time * time_s_scaled
    work["_time_hours"] = work[time_col_s] / 3600.0

    scored = work.sort_values("_tradeoff_score", ascending=True).reset_index(drop=True)
    best = scored.iloc[0]

    created_fig = False
    if ax_pareto is None or ax_score is None:
        fig, (ax_pareto, ax_score) = plt.subplots(1, 2, figsize=(12, 4.8))
        created_fig = True

    # (1) time vs loss with best
    ax_pareto.scatter(scored["_time_hours"].values, scored[loss_col].values, s=25)
    ax_pareto.scatter([best["_time_hours"]], [best[loss_col]], marker="*", s=180)
    ax_pareto.set_xlabel("Training time [hours]")
    ax_pareto.set_ylabel("Best validation loss")
    ax_pareto.set_title(f"Tradeoff (wL={w_loss:.2f}, wT={w_time:.2f}, {scaling})")
    ax_pareto.grid(True)

    # (2) score vs rank
    ax_score.plot(np.arange(len(scored)), scored["_tradeoff_score"].values, marker="o", linewidth=1)
    ax_score.set_xlabel("Rank (lower is better)")
    ax_score.set_ylabel("Tradeoff score")
    ax_score.set_title("Tradeoff score by rank")
    ax_score.grid(True)

    if created_fig and save:
        fig.tight_layout()
        p = outdir / f"{filename_prefix}_summary.pdf"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        scored.to_csv(outdir / f"{filename_prefix}_scored_table.csv", index=False)

    return scored, best


def scatter3d_hparam_grid_colored(
    df: pd.DataFrame,
    triples: list[tuple[str, str, str]],
    color_col: str,
    outdir: str | Path,
    filename_prefix: str = "scatter3d",
    dpi: int = 200,
) -> list[Path]:
    """
    Runs scatter3d_inputs_with_output_colorbar for multiple (x,y,z) triples.
    Returns list of saved paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for (x, y, z) in triples:
        fname = f"{filename_prefix}_{x}_{y}_{z}_c_{color_col}.pdf"
        title = f"3D scatter ({x}, {y}, {z}) colored by {color_col}"
        p = scatter3d_inputs_with_output_colorbar(
            df=df,
            x_col=x, y_col=y, z_col=z,
            c_col=color_col,
            outdir=outdir,
            filename=fname,
            title=title,
            dpi=dpi,
        )
        saved_paths.append(p)
    return saved_paths


def pareto_front_loss_time(
    df: pd.DataFrame,
    loss_col: str = "best_val_loss",
    time_col_s: str = "_time_seconds",
    run_col: str = "run",
    outdir: str | Path = "run_analysis_out",
    filename: str = "pareto_front_loss_time.pdf",
    dpi: int = 200,
    ax: Optional[plt.Axes] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Returns Pareto-front rows minimizing (loss, time).
    If ax is provided, plots into that axis and does NOT create/close figure.
    If save=True and ax is None, saves a single PDF.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    for c in [loss_col, time_col_s]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[loss_col, time_col_s]).copy()
    if work.empty:
        raise ValueError("No valid rows for Pareto front.")

    work["_time_hours"] = work[time_col_s] / 3600.0
    work = work.sort_values([loss_col, time_col_s], ascending=[True, True]).reset_index(drop=True)

    pareto_idx = []
    best_time = np.inf
    for i, row in work.iterrows():
        t = row[time_col_s]
        if t < best_time:
            pareto_idx.append(i)
            best_time = t
    pareto = work.loc[pareto_idx].copy()

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        created_fig = True

    ax.scatter(work["_time_hours"].values, work[loss_col].values, s=25)
    ax.scatter(pareto["_time_hours"].values, pareto[loss_col].values, s=60)
    ax.set_xlabel("Training time [hours]")
    ax.set_ylabel("Best validation loss")
    ax.set_title("Pareto front: minimize (loss, time)")
    ax.grid(True)

    if created_fig and save:
        fig.tight_layout()
        p = outdir / filename
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        pareto.to_csv(outdir / "pareto_front_table.csv", index=False)

    return pareto


def scatter2d_on_ax(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    cols = [x_col, y_col]
    work = df.copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    if work.empty:
        ax.set_title(title or f"{x_col} vs {y_col} (no data)")
        ax.axis("off")
        return

    ax.scatter(work[x_col].values, work[y_col].values)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"{x_col} vs {y_col}")
    ax.grid(True)


def save_scatter2d_grid_pdf(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, str]],  # (x_col, y_col, title)
    outdir: str | Path,
    filename: str = "scatters_grid.pdf",
    ncols: int = 3,
    dpi: int = 200,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = len(pairs)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (x, y, t) in enumerate(pairs):
        scatter2d_on_ax(df, x, y, ax=axes[i], title=t)

    # spegni gli assi inutilizzati
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    outpath = outdir / filename
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def scatter3d_on_ax(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    c_col: str,
    ax,
    title: Optional[str] = None,
    elev: float = 20,
    azim: float = -60,
    s: float = 30,
):
    cols = [x_col, y_col, z_col, c_col]
    work = df.copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    if len(work) < 3:
        ax.set_title(title or "3D scatter (no data)")
        ax.axis("off")
        return None

    sc = ax.scatter(
        work[x_col].to_numpy(),
        work[y_col].to_numpy(),
        work[z_col].to_numpy(),
        c=work[c_col].to_numpy(),
        s=s
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title or f"({x_col},{y_col},{z_col}) by {c_col}")
    ax.view_init(elev=elev, azim=azim)
    return sc


def save_scatter3d_grid_pdf(
    df: pd.DataFrame,
    triples: list[tuple[str, str, str, str]],  # (x,y,z,c)
    outdir: str | Path,
    filename: str = "scatters_3d_grid.pdf",
    ncols: int = 2,
    dpi: int = 200,
    elev: float = 20,
    azim: float = -60,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = len(triples)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(6.0 * ncols, 5.2 * nrows))
    axes = []
    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        axes.append(ax)

    mappable = None
    for i, (x, y, z, c) in enumerate(triples):
        sc = scatter3d_on_ax(
            df=df, x_col=x, y_col=y, z_col=z, c_col=c,
            ax=axes[i],
            title=f"({x},{y},{z}) colored by {c}",
            elev=elev, azim=azim
        )
        if mappable is None and sc is not None:
            mappable = sc

    # spegni assi inutilizzati
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # colorbar condivisa (se abbiamo almeno un plot valido)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.75, pad=0.02)
        cbar.set_label(triples[0][3])  # label = c_col del primo, se li tieni coerenti

    fig.tight_layout()
    outpath = outdir / filename
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def plot_one_param_slices_on_axes(
    df: pd.DataFrame,
    param: str,
    loss_ax: plt.Axes,
    time_ax: plt.Axes,
    loss_col: str = "best_val_loss",
    time_col_s: str = "train_time_seconds",
    hyperparams: Tuple[str, ...] = ("n_layer", "n_head", "n_embd", "batch_size"),
    min_points_per_slice: int = 2,
):
    fixed = [p for p in hyperparams if p != param]
    work = df.copy()
    for c in [*hyperparams, loss_col, time_col_s]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[param, *fixed, loss_col, time_col_s]).copy()
    if work.empty:
        loss_ax.set_title(f"{param}: no data")
        time_ax.set_title(f"{param}: no data")
        return

    def _label(fixed_vals):
        if not isinstance(fixed_vals, tuple):
            fixed_vals = (fixed_vals,)
        return ", ".join(f"{k}={int(v) if float(v).is_integer() else v}" for k, v in zip(fixed, fixed_vals))

    # LOSS
    for fixed_vals, g in work.groupby(fixed, dropna=False):
        g = g.sort_values(param)
        if len(g) < min_points_per_slice:
            continue
        loss_ax.plot(g[param].values, g[loss_col].values, marker="o", linewidth=1.0, label=_label(fixed_vals))

    loss_ax.set_xlabel(param)
    loss_ax.set_ylabel("Best validation loss")
    loss_ax.set_title(f"{param}: loss (others fixed)")
    loss_ax.grid(True)

    # TIME
    for fixed_vals, g in work.groupby(fixed, dropna=False):
        g = g.sort_values(param)
        if len(g) < min_points_per_slice:
            continue
        time_ax.plot(g[param].values, (g[time_col_s].values / 3600.0), marker="o", linewidth=1.0, label=_label(fixed_vals))

    time_ax.set_xlabel(param)
    time_ax.set_ylabel("Training time [hours]")
    time_ax.set_title(f"{param}: time (others fixed)")
    time_ax.grid(True)


def save_slices_all_params_pdf(
    df: pd.DataFrame,
    params: list[str],
    outdir: str | Path,
    filename: str = "slices_all_params.pdf",
    loss_col: str = "best_val_loss",
    time_col_s: str = "train_time_seconds",
    hyperparams: Tuple[str, ...] = ("n_layer", "n_head", "n_embd", "batch_size"),
    min_points_per_slice: int = 2,
    dpi: int = 200,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ncols = len(params)
    fig, axes = plt.subplots(2, ncols, figsize=(4.8 * ncols, 7.2), squeeze=False)

    for j, p in enumerate(params):
        plot_one_param_slices_on_axes(
            df=df,
            param=p,
            loss_ax=axes[0, j],
            time_ax=axes[1, j],
            loss_col=loss_col,
            time_col_s=time_col_s,
            hyperparams=hyperparams,
            min_points_per_slice=min_points_per_slice,
        )

    fig.tight_layout()
    outpath = outdir / filename
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def save_bars_loss_time_one_pdf(
    core_sorted_loss: pd.DataFrame,
    core_sorted_time: pd.DataFrame,
    run_col: str,
    loss_col: str,
    time_hours_col: str,
    outdir: str | Path,
    filename: str = "bars_loss_time.pdf",
    dpi: int = 200,
) -> Path:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(core_sorted_loss[run_col].astype(str), core_sorted_loss[loss_col].values)
    ax1.set_title("Best validation loss per run (lower is better)")
    ax1.set_ylabel("Best validation loss")
    ax1.tick_params(axis="x", rotation=60, labelsize=8)

    ax2.bar(core_sorted_time[run_col].astype(str), core_sorted_time[time_hours_col].values)
    ax2.set_title("Training time per run")
    ax2.set_ylabel("Training time [hours]")
    ax2.tick_params(axis="x", rotation=60, labelsize=8)

    fig.tight_layout()
    p = Path(outdir) / filename
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p



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
    time_sec_col: str = "train_time_seconds",
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
    """for hp in ["n_layer", "n_head", "n_embd", "batch_size", 
               "smoothness", "regression_mode", "activation_function", ]:
        if hp not in df.columns:
            df[hp] = df[run_col].apply(lambda r: infer_hparams_from_run_name(str(r)).get(hp))"""

    df = ensure_numeric(df, ["_time_seconds", val_loss_col])    #"n_layer", "n_head", "n_embd", "batch_size", 

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

    """# Plot 3: Scatter time vs loss
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
            saved.append(p)"""

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


    # =========================
    # Compact reporting (few PDFs)
    # =========================

    # (A) One PDF: tradeoff summary (2 subplots)
    scored, best_tradeoff = pick_best_tradeoff_run(
        df=cleaned,
        run_col=run_col,
        loss_col=val_loss_col,
        time_col_s="_time_seconds",
        w_loss=0.75,
        w_time=0.25,
        scaling="minmax",
        outdir=outdir,
        filename_prefix="tradeoff_w075_w025",
        save=True,   # saves ONE PDF: tradeoff_w075_w025_summary.pdf
    )
    saved.append(Path(outdir) / "tradeoff_w075_w025_summary.pdf")
    saved.append(Path(outdir) / "tradeoff_w075_w025_scored_table.csv")

    # (B) One PDF: Pareto
    pareto = pareto_front_loss_time(
        df=cleaned,
        loss_col=val_loss_col,
        time_col_s="_time_seconds",
        run_col=run_col,
        outdir=outdir,
        filename="pareto_front_loss_time.pdf",
        save=True
    )
    saved.append(Path(outdir) / "pareto_front_loss_time.pdf")
    saved.append(Path(outdir) / "pareto_front_table.csv")


    # --- 3D scatters colored by metrics (if present) ---
    paths = scatter3d_hparam_grid_colored(
        df=cleaned,
        triples=[
            ("n_layer", "n_head", "n_embd"),
            ("n_layer", "n_head", "batch_size"),
            ("n_head", "n_embd", "batch_size"),
        ],
        color_col=val_loss_col,
        outdir=outdir,
        filename_prefix="scatter3d_val_loss",
    )
    saved.extend(paths)

    paths = scatter3d_hparam_grid_colored(
        df=cleaned,
        triples=[
            ("n_layer", "n_head", "n_embd"),
            ("n_layer", "n_head", "batch_size"),
            ("n_head", "n_embd", "batch_size"),
        ],
        color_col="_time_seconds",
        outdir=outdir,
        filename_prefix="scatter3d_time_seconds",
    )
    saved.extend(paths)

    pairs = [("_time_hours", val_loss_col, "Time [h] vs Val loss")]
    for hp in ["n_layer", "n_head", "n_embd", "batch_size"]:
        if hp in cleaned.columns and cleaned[hp].notna().sum() >= 2:
            pairs.append((hp, val_loss_col, f"{hp} vs val loss"))
            pairs.append((hp, "_time_hours", f"{hp} vs time [h]"))

    p_grid = save_scatter2d_grid_pdf(
        df=cleaned,
        pairs=pairs,
        outdir=outdir,
        filename="scatters_2d_grid.pdf",
        ncols=3,
    )
    saved.append(p_grid)

    """triples = [
        ("n_layer", "n_head", "n_embd", val_loss_col),
        ("n_layer", "n_head", "batch_size", val_loss_col),
        ("n_head", "n_embd", "batch_size", val_loss_col),
        ("n_layer", "n_head", "n_embd", "_time_hours"),
        ("n_layer", "n_head", "batch_size", "_time_hours"),
        ("n_head", "n_embd", "batch_size", "_time_hours"),
    ]
    p3dgrid = save_scatter3d_grid_pdf(
        df=cleaned,
        triples=triples,
        outdir=outdir,
        filename="scatters_3d_grid_color_loss.pdf",
        ncols=2,
    )
    saved.append(p3dgrid)"""

    cleaned_for_slices = cleaned.rename(columns={"_time_seconds": "train_time_seconds"})

    p_slices = save_slices_all_params_pdf(
        df=cleaned_for_slices,
        params=["n_layer", "n_head", "n_embd", "batch_size"],
        outdir=outdir,
        filename="slices_all_params.pdf",
        loss_col=val_loss_col,
        time_col_s="train_time_seconds",
        min_points_per_slice=2,
    )
    saved.append(p_slices)


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

    """# Example: layers effect with heads/embd/batch fixed
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
        )"""


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
python plot_sweep_analisys.py --csv <name_sweep>.csv --base runs/<sweep_folder>
"""

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sweep_results.csv", help="Path to runs CSV")
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
