from __future__ import annotations

import itertools
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import csv


@dataclass
class SweepResult:
    rows: List[Dict[str, Any]]
    output_csv: Optional[Path] = None


class Sweeper:
    """
    Unified sweep runner supporting:
      - grid
      - random (with optional distributions)
      - optuna (with optional pruning)
      - successive_halving (Hyperband-lite)

    Expected cfg structure:

    cfg["sweep"] = {
        "type": "grid" | "random" | "optuna" | "successive_halving",
        "params": { ... },              # discrete choices or ranges depending on method
        "distributions": { ... },       # for random/optuna continuous sampling (optional)
        ... method-specific keys ...
    }

    Minimal examples:

    Grid:
      sweep:
        type: grid
        params:
          n_layer: [4, 6, 8]
          activation_function: ["gelu", "silu"]

    Random:
      sweep:
        type: random
        n_trials: 50
        params:
          n_layer: [4, 6, 8]   # discrete
          activation_function: ["gelu", "silu"]
        distributions:
          lr: {type: loguniform, low: 1e-5, high: 3e-3}
          weight_decay: {type: loguniform, low: 1e-6, high: 1e-2}
          dropout: {type: uniform, low: 0.0, high: 0.3}

    Optuna:
      sweep:
        type: optuna
        n_trials: 50
        params: {activation_function: ["gelu","silu"], n_layer: [4,6,8]}
        distributions: {lr: {type: loguniform, low: 1e-5, high: 3e-3}}
        pruning: true

    Successive Halving:
      sweep:
        type: successive_halving
        n_trials: 60
        eta: 3
        budgets: [5, 15, 50]           # epochs per rung
        budget_key: "training.max_epochs"
        params: {...}
        distributions: {...}
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        load_datasets_fn,
        run_single_experiment_fn,
        analyze_runs_csv_fn,
    ):
        self.base_cfg = deepcopy(cfg)
        self.load_datasets = load_datasets_fn
        self.run_single_experiment = run_single_experiment_fn
        self.analyze_runs_csv = analyze_runs_csv_fn

        self.sweep_cfg = self.base_cfg.get("sweep", {})
        if not self.sweep_cfg:
            raise ValueError("No cfg['sweep'] block found.")

        self.sweep_type = str(self.sweep_cfg.get("type", "grid")).lower()
        self.params: Dict[str, Any] = dict(self.sweep_cfg.get("params", {}))
        self.dists: Dict[str, Any] = dict(self.sweep_cfg.get("distributions", {}))

        exp = self.base_cfg.get("experiment", {})
        self.output_root = Path(exp.get("output_root", "out"))
        self.output_sweep = exp.get("output_sweep", "sweep")
        self.root = self.output_root / self.output_sweep
        self.root.mkdir(parents=True, exist_ok=True)
        self.sweep_path = self.sweep_cfg.get("path", "sweep_results.csv")

        self.seed = int(self.sweep_cfg.get("seed", exp.get("seed", 0)))
        self.rng = random.Random(self.seed)

        # Preload datasets once (your original behavior)
        self.train_ds, self.val_ds, self.test_ds = self.load_datasets(self.base_cfg["data"])

    # ----------------------------
    # Public API
    # ----------------------------
    def run(self) -> SweepResult:
        if self.sweep_type == "grid":
            rows = self._run_grid()
        elif self.sweep_type == "random":
            rows = self._run_random()
        elif self.sweep_type == "optuna":
            rows = self._run_optuna()
        elif self.sweep_type in ("successive_halving", "sh"):
            rows = self._run_successive_halving()
        else:
            raise ValueError(f"Unknown sweep type: {self.sweep_type}")

        out_csv = self._write_results_csv(rows)

        try:
            self.analyze_runs_csv(
                    csv_path=self.sweep_pth,
                    outdir="sweep_analysis_out",
                    base=self.root,
                    run_col="run",
                    val_loss_col="best_val_loss",
                    time_col="train_time_seconds",
                    time_is_seconds=True,
                )
        except Exception as e:
            print(f"[main] Warning: failed to analyze sweep results: {e}")

        return SweepResult(rows=rows, output_csv=out_csv)

    # ----------------------------
    # Core helpers
    # ----------------------------
    def _apply_overrides(self, cfg_run: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        # Same policy you used, centralized
        for k, v in overrides.items():
            if k in cfg_run.get("model", {}):
                cfg_run["model"][k] = v
            elif k in cfg_run.get("training", {}):
                cfg_run["training"][k] = v
            elif k in cfg_run.get("scheduled_sampling", {}):
                cfg_run["scheduled_sampling"][k] = v
            else:
                cfg_run.setdefault("training", {})[k] = v

    def _set_by_dotted_key(self, cfg_run: Dict[str, Any], dotted: str, value: Any) -> None:
        # e.g. "training.max_epochs"
        parts = dotted.split(".")
        cur = cfg_run
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def _make_run_name(self, overrides: Dict[str, Any], prefix: str = "") -> str:
        items = []
        for k in sorted(overrides.keys()):
            v = overrides[k]
            if isinstance(v, float):
                items.append(f"{k}{v:.3g}")
            else:
                items.append(f"{k}{v}")
        name = "_".join(items)
        return f"{prefix}{name}" if prefix else name

    def _sample_from_dist(self, dist: Dict[str, Any]) -> float:
        t = str(dist["type"]).lower()
        low, high = float(dist["low"]), float(dist["high"])
        if t == "uniform":
            return self.rng.uniform(low, high)
        if t == "loguniform":
            return math.exp(self.rng.uniform(math.log(low), math.log(high)))
        raise ValueError(f"Unknown dist type: {t}")

    def _iter_random_overrides(self, n_trials: int) -> Iterator[Dict[str, Any]]:
        keys = list(self.params.keys())
        dist_keys = list(self.dists.keys())

        for _ in range(n_trials):
            overrides: Dict[str, Any] = {}
            # discrete/categorical from params (lists)
            for k in keys:
                choices = self.params[k]
                if not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(f"Random sweep expects params['{k}'] to be a non-empty list.")
                overrides[k] = self.rng.choice(choices)

            # continuous from distributions
            for k in dist_keys:
                overrides[k] = self._sample_from_dist(self.dists[k])

            yield overrides

    def _run_one(self, overrides: Dict[str, Any], run_name: str) -> Dict[str, Any]:
        cfg_run = deepcopy(self.base_cfg)
        cfg_run["experiment"]["name"] = run_name

        self._apply_overrides(cfg_run, overrides)

        run_dir = self.root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss, train_time_s, test_loss, num_params = self.run_single_experiment(
            cfg_run,
            self.train_ds,
            self.val_ds,
            self.test_ds,
            run_dir,
            run_name,
        )

        row = {
            "run": run_name,
            "best_val_loss": best_val_loss,
            "train_time_seconds": train_time_s,
            "test_loss": test_loss,
            "num_params": num_params,
        }
        row.update(overrides)
        return row

    def _write_results_csv(self, rows: List[Dict[str, Any]]) -> Optional[Path]:
        if not rows:
            return None

        out_csv = self.root / self.sweep_path
        # union of keys for header
        keys = sorted({k for r in rows for k in r.keys()})

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        
        print(f"[main] Saved sweep summary to {out_csv}")
        return out_csv

    # ----------------------------
    # Sweep methods
    # ----------------------------
    def _run_grid(self) -> List[Dict[str, Any]]:
        if not self.params:
            raise ValueError("Grid sweep requires sweep.params")

        keys, values = zip(*self.params.items())
        for k, v in self.params.items():
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(f"Grid sweep expects params['{k}'] to be a non-empty list.")

        rows: List[Dict[str, Any]] = []
        for i, combo in enumerate(itertools.product(*values)):
            overrides = dict(zip(keys, combo))
            name = self._make_run_name(overrides)
            print(f"[sweep:grid] {i}: {name}")
            rows.append(self._run_one(overrides, name))
        return rows

    def _run_random(self) -> List[Dict[str, Any]]:
        n_trials = int(self.sweep_cfg.get("n_trials", 50))
        if not self.params and not self.dists:
            raise ValueError("Random sweep needs at least sweep.params or sweep.distributions")

        rows: List[Dict[str, Any]] = []
        for i, overrides in enumerate(self._iter_random_overrides(n_trials)):
            name = self._make_run_name(overrides)
            print(f"[sweep:random] {i}: {name}")
            rows.append(self._run_one(overrides, name))
        return rows

    def _run_optuna(self) -> List[Dict[str, Any]]:
        try:
            import optuna
        except Exception as e:
            raise ImportError("Optuna sweep requested but optuna is not installed. pip install optuna") from e

        n_trials = int(self.sweep_cfg.get("n_trials", 50))
        pruning = bool(self.sweep_cfg.get("pruning", True))

        # sampler/pruner
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2) if pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        rows: List[Dict[str, Any]] = []

        def objective(trial: "optuna.Trial") -> float:
            overrides: Dict[str, Any] = {}

            # discrete from params as categorical
            for k, choices in self.params.items():
                if not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(f"Optuna sweep expects params['{k}'] to be a non-empty list.")
                overrides[k] = trial.suggest_categorical(k, choices)

            # continuous from distributions
            for k, dist in self.dists.items():
                t = str(dist["type"]).lower()
                low, high = float(dist["low"]), float(dist["high"])
                if t == "uniform":
                    overrides[k] = trial.suggest_float(k, low, high)
                elif t == "loguniform":
                    overrides[k] = trial.suggest_float(k, low, high, log=True)
                else:
                    raise ValueError(f"Unknown dist type for {k}: {t}")

            name = self._make_run_name(overrides)
            cfg_run = deepcopy(self.base_cfg)
            cfg_run["experiment"]["name"] = name
            self._apply_overrides(cfg_run, overrides)

            run_dir = self.root / name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Optional pruning hook: requires your run_single_experiment to call report_cb(epoch, val_loss)
            def report_cb(epoch: int, val_loss: float):
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if pruning:
                best_val_loss, train_time_s, test_loss, num_params = self.run_single_experiment(
                    cfg_run, self.train_ds, self.val_ds, self.test_ds, run_dir, name, report_cb=report_cb
                )
            else:
                best_val_loss, train_time_s, test_loss, num_params = self.run_single_experiment(
                    cfg_run, self.train_ds, self.val_ds, self.test_ds, run_dir, name
                )

            row = {
                "run": name,
                "best_val_loss": best_val_loss,
                "train_time_seconds": train_time_s,
                "test_loss": test_loss,
                "num_params": num_params,
            }
            row.update(overrides)
            rows.append(row)

            # return objective
            return float(best_val_loss)

        study.optimize(objective, n_trials=n_trials)

        # rows already collected
        return rows

    def _run_successive_halving(self) -> List[Dict[str, Any]]:
        n_trials = int(self.sweep_cfg.get("n_trials", 60))
        eta = int(self.sweep_cfg.get("eta", 3))
        budgets = list(self.sweep_cfg.get("budgets", [5, 15, 50]))
        budget_key = str(self.sweep_cfg.get("budget_key", "training.max_epochs"))

        if eta < 2:
            raise ValueError("successive_halving requires eta >= 2")
        if not budgets:
            raise ValueError("successive_halving requires non-empty budgets list")

        # initial candidate pool from random sampling
        candidates = list(self._iter_random_overrides(n_trials))

        rows: List[Dict[str, Any]] = []
        survivors = candidates

        for r, budget in enumerate(budgets):
            scored: List[Tuple[float, Dict[str, Any]]] = []

            for i, overrides in enumerate(survivors):
                prefix = f"r{r}_b{budget}_"
                name = self._make_run_name(overrides, prefix=prefix)

                cfg_run = deepcopy(self.base_cfg)
                cfg_run["experiment"]["name"] = name
                self._apply_overrides(cfg_run, overrides)
                self._set_by_dotted_key(cfg_run, budget_key, int(budget))

                run_dir = self.root / name
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"[sweep:sh] round={r} budget={budget} i={i} name={name}")

                best_val_loss, train_time_s, test_loss, num_params = self.run_single_experiment(
                    cfg_run, self.train_ds, self.val_ds, self.test_ds, run_dir, name
                )

                row = {
                    "run": name,
                    "round": r,
                    "budget": budget,
                    "budget_key": budget_key,
                    "best_val_loss": best_val_loss,
                    "train_time_seconds": train_time_s,
                    "test_loss": test_loss,
                    "num_params": num_params,
                }
                row.update(overrides)
                rows.append(row)

                scored.append((float(best_val_loss), overrides))

            scored.sort(key=lambda x: x[0])
            k = max(1, len(scored) // eta)
            survivors = [ov for _, ov in scored[:k]]

        return rows

