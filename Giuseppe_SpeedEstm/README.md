# Speed Estimator — Zerostep Transformer

Modular training stack for a BLDC speed-estimation task using a causal Transformer with rollout (no teacher forcing). The codebase is split by concerns so you can start small and scale to deeper models, richer losses, and multiple datasets without rewiring everything.

---

## Repository layout

```
project-root/
  configs/
    default.yaml
    dataset/real_bldc.yaml
    model/zerostep_small.yaml
    train/baseline.yaml
    train/smoothing.yaml
  data/
    raw/...
    processed/...
  src/
    datatypes.py
    datasets/
      bldc_csv.py
      transforms.py
      registry.py
    models/
      transformer_zerostep.py
      heads.py
      registry.py
    losses/
      mse.py
      smoothness.py
      composite.py
    optim/
      optimizers.py
      schedulers.py
      factory.py
    engine/
      trainer.py
      evaluator.py
      callbacks.py
      metrics.py
      seed.py
      checkpoint.py
    utils/
      io.py
      logging.py
      plotting.py
      timers.py
  scripts/
    train.py
    evaluate.py
    plot_runs.py
  runs/
  tests/
  pyproject.toml
  README.md
  requirements.txt
```

- **configs/** YAML configs layered by domain (data, model, train). No magic numbers in code.  
- **src/** implementation modules, organized by concern. Public construction happens via registries/factories.  
- **scripts/** thin CLI entrypoints that glue configs and factories together.  
- **runs/** experiment artifacts (timestamped folders with cfg, checkpoints, metrics, plots).  
- **tests/** lightweight unit tests for critical invariants.  

---

## Installation

### 1) Python
Use Python 3.10–3.12.

### 2) Create environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3) Install (CPU)
```bash
pip install -e .
```

### 4) Install (GPU)
Install a CUDA-compatible PyTorch wheel first (matching your CUDA/OS), then:
```bash
pip install -e .
```

### 5) Optional dev tools
```bash
pip install .[dev]
```

---

## Datasets

Expected input is a folder of CSV files with currents, voltages, and speed. Configure paths and normalization in `configs/dataset/*.yaml`.

- **Columns** (example): `ia, ib, va, vb, omega`  
- **Windowing**: sequences of length `seq_len` with the **last channel** reserved for `last_omega` (the previous prediction during rollout).  
- **Normalization**:  
  - `fixed_range` maps to [0,1] using per-channel ranges.  
  - `zscore` uses mean/std computed from the training split only.  

To add a new dataset, implement `BLDCSequenceDataset` in `src/datasets/<name>.py` and register it in `src/datasets/registry.py`.

---

## Configuration

Configs are merged in order: `configs/default.yaml` → domain overrides → CLI overrides.

Examples:
```bash
# Baseline zero-step training
speed-train --config configs/train/baseline.yaml

# Smoothing variant and a deeper model
speed-train --config configs/train/smoothing.yaml \
            --override model.n_layer=8 model.d_model=256 loss.components.smoothness.weight=0.2

# Evaluate a checkpoint
speed-eval --ckpt runs/2025-11-11_10-03_bldc_zerostep_baseline/ckpt_best.pt --split test
```

Common fields:
```yaml
seed: 1337
device: "cuda"
exp_name: "bldc_zerostep"

data:
  root: "data/processed"
  split:
    train: ["2024-*.csv"]
    val:   ["2025-*.csv"]
  seq_len: 128
  batch_size: 64
  num_workers: 8
  normalize:
    method: "fixed_range"
    ranges:
      ia: [-20, 20]
      ib: [-20, 20]
      va: [0, 60]
      vb: [0, 60]
      omega: [0, 8000]
  inject_last_channel: true

model:
  name: "gpt_zerostep"   # registered in models/registry.py
  d_model: 128
  n_layer: 4
  n_head: 4
  dropout: 0.1

train:
  epochs: 200
  grad_clip: 1.0
  mixed_precision: "fp16"   # "off"|"fp16"|"bf16"
  patience: 20
  val_every: 1
  save_every: 5

optim:
  name: "adamw"
  lr: 3e-4
  weight_decay: 0.01
  scheduler:
    name: "warmup_cosine"
    warmup_steps: 1000
    max_steps: 60000

loss:
  components:
    mse: { weight: 1.0 }
    smoothness: { weight: 0.1, diff_order: 1 }
```

---

## Training

```bash
speed-train --config configs/train/baseline.yaml
```

What happens:
- Builds dataset/dataloaders from the `datasets` registry.
- Builds the model from the `models` registry.
- Assembles composite loss from `losses`.
- Configures optimizer and LR scheduler from `optim`.
- Runs a rollout training loop that **feeds previous prediction** into the last input channel.
- Logs per-epoch metrics, saves the best checkpoint and the last checkpoint.
- Creates a run folder: `runs/{YYYY-MM-DD_HH-MM}_{exp_name}/`.

Artifacts inside a run folder:
```
cfg.yaml                # merged config used
history.csv             # epoch, train_loss, val_loss, mse, smoothness, lr
ckpt_best.pt            # best-on-val
ckpt_last.pt            # last epoch
metrics_val.csv         # validation metrics at best
curves.png              # loss curves
logs.jsonl              # structured logs
```

---

## Evaluation

```bash
speed-eval --ckpt runs/<timestamp>_<exp>/ckpt_best.pt --split test
```

- Rebuilds the model from the stored config.
- Runs deterministic rollout that mirrors training.
- Writes `metrics_test.csv` next to the checkpoint.

---

## Plotting training curves

```bash
speed-plot runs/2025-11-11_10-03_bldc_zerostep_baseline runs/2025-11-11_14-22_deeper
```

Generates overlay plots and a summary CSV of final metrics.

---

## Extending the project

- **New dataset**: implement in `src/datasets/`, add to `registry.py`, add a YAML under `configs/dataset/`.  
- **New model**: implement in `src/models/`, register it, add a model YAML.  
- **New loss**: create a file in `src/losses/` and wire it in `losses/composite.py`.  
- **Curriculum or callbacks**: add to `src/engine/callbacks.py` and toggle via config.  

---

## Reproducibility

- All knobs live in YAML; each run folder stores the exact `cfg.yaml`.  
- `engine/seed.py` seeds Python, NumPy, and PyTorch. Determinism can be toggled if you need cuDNN speed.  

---

## Testing

```bash
pytest -q
```

- `test_dataset.py`: shapes, normalization bounds.  
- `test_rollout.py`: verifies rollout uses previous predictions.  
- `test_scheduler.py`: warmup/cosine schedule endpoints.  

---

## License

MIT (or your choice). See `LICENSE`.
