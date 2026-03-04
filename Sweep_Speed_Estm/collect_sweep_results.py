import os
import json
from typing import Any, Dict, List, Optional

import pandas as pd


def _get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely fetch nested values like ['a','b','c'] from dict d."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_sweep_results(
    folder_name: str,
    csv_name: str,
    runs_root: str = "runs",
    summary_filename: str = "summary.json",
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Collect results from runs/{folder_name}/**/summary.json and export to CSV.

    Extracted columns:
      - run (exp_name if present, else folder name)
      - best_val_loss
      - test_loss
      - train_time_seconds
      - num_params

    Args:
        folder_name: e.g. "sweeps_4"
        csv_name: e.g. "sweep_results" (will save as sweep_results.csv)
        runs_root: defaults to "runs"
        summary_filename: defaults to "summary.json"
        recursive: if True, searches all subdirectories for summary.json;
                   if False, only checks immediate subfolders.

    Returns:
        pandas DataFrame with one row per run.
    """
    base_dir = os.path.join(runs_root, folder_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    rows: List[Dict[str, Any]] = []

    def handle_summary(summary_path: str):
        run_dir = os.path.dirname(summary_path)
        run_folder = os.path.basename(run_dir)

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            # Keep going, but record an error row if you want
            rows.append(
                {
                    "run": run_folder,
                    "best_val_loss": None,
                    "test_loss": None,
                    "train_time_seconds": None,
                    "num_params": None,
                    "summary_path": summary_path,
                    "error": f"Failed to read/parse JSON: {e}",
                }
            )
            return

        # Common variants for keys across projects
        exp_name = data.get("exp_name") or data.get("run") or data.get("name") or run_folder

        best_val_loss = (
            data.get("best_val_loss")
            if "best_val_loss" in data
            else _get_nested(data, ["metrics", "best_val_loss"])
        )

        test_loss = (
            data.get("test_loss")
            if "test_loss" in data
            else _get_nested(data, ["metrics", "test_loss"])
        )

        train_time_seconds = (
            data.get("train_time_seconds")
            if "train_time_seconds" in data
            else _get_nested(data, ["timing", "train_time_seconds"])
        )

        num_params = (
            data.get("num_params")
            if "num_params" in data
            else _get_nested(data, ["model", "num_params"])
        )

        rows.append(
            {
                "run": exp_name,
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
                "train_time_seconds": train_time_seconds,
                "num_params": num_params,
                "summary_path": summary_path,
                "error": None,
            }
        )

    if recursive:
        for root, _, files in os.walk(base_dir):
            if summary_filename in files:
                handle_summary(os.path.join(root, summary_filename))
    else:
        for sub in os.listdir(base_dir):
            sub_path = os.path.join(base_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            summary_path = os.path.join(sub_path, summary_filename)
            if os.path.exists(summary_path):
                handle_summary(summary_path)

    df = pd.DataFrame(rows)

    # Helpful typing/conversions
    for col in ["best_val_loss", "test_loss", "train_time_seconds", "num_params"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort with best_val_loss if available
    if "best_val_loss" in df.columns and df["best_val_loss"].notna().any():
        df = df.sort_values(["best_val_loss", "run"], ascending=[True, True])
    else:
        df = df.sort_values(["run"], ascending=True)

    out_path = os.path.join(base_dir, f"{csv_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} runs to {out_path}")
    return df


if __name__ == "__main__":
    # Example:
    # python collect_results.py
    df = collect_sweep_results(folder_name="sweeps_4", csv_name="sweep_results")
    print(df.head(20).to_string(index=False))
