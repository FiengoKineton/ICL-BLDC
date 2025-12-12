# plot_all.py           | SET (leave it as it is)

from pathlib import Path
import subprocess

def run_plots_for_all_sweeps(print_flag: bool = False) -> None:
    """
    For each subfolder in runs/sweeps, run:
      - python plot_testing.py  --dir runs/sweeps/{name_sweep}
      - python plot_training.py --dir runs/sweeps/{name_sweep}
    """
    repo_root = Path(__file__).resolve().parent
    sweeps_root = repo_root / "runs" / "sweeps"

    if not sweeps_root.exists():
        raise FileNotFoundError(f"sweeps root not found: {sweeps_root}")

    sweep_dirs = sorted(p for p in sweeps_root.iterdir() if p.is_dir())
    if not sweep_dirs:
        if print_flag: print(f"No sweep folders found in {sweeps_root}")
        return

    for sweep_dir in sweep_dirs:
        name_sweep = sweep_dir.name

        # what goes into --dir (relative path from repo root, like runs/sweeps/xxx)
        rel_dir = sweep_dir.relative_to(repo_root)
        dir_arg = str(rel_dir)

        if print_flag: print(f"\n=== Sweep: {name_sweep} ===")
        if print_flag: print(f"Run dir (for --dir): {dir_arg}")

        for script in ("plot_testing.py", "plot_training.py"):
            cmd = ["python", script, "--dir", dir_arg]
            if print_flag: print(f"Running: {' '.join(cmd)}")

            try:
                subprocess.run(cmd, check=True, cwd=repo_root)
            except subprocess.CalledProcessError as e:
                print(
                    f"[WARNING] {script} failed for sweep '{name_sweep}' "
                    f"with return code {e.returncode}"
                )
                # move on to the next script / sweep


if __name__ == "__main__":
    run_plots_for_all_sweeps(False)

# 0.00069473 — n_layer8_n_head2_n_embd32_batch_size64 (not finished)