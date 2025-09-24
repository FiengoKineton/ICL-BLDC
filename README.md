# In-Context Learning for Zero-Shot Speed Estimation of BLDC motors

This repository contains the Python and Matlab code to reproduce the results of the paper [In-Context Learning for Zero-Shot Speed Estimation of BLDC motors](https://arxiv.org/abs/2504.00673)
by Alessandro Colombo, Riccardo Busetto, Valentina Breschi, Marco Forgione, Dario Piga, and Simone Formentin

We propose a learning framework in which a meta model is trained to describe the *class* of BLDC motors. It receives as input a windows the last __H__ samples of the voltages applied to the motor and the measured currents, and estimates the current motor speed.
By training the model on a set of simulated experiments, in which the motor parameters were randomly perturbed, the model is able to comprehend the intrinsic parameters of the system and adapt to different motor configurations.

# Main files

## Simulation

In the *matlab_simulator* folder:
 * __BLDC_simulator.slx__ contains the controller simulator
 * __BLDC_simulator_init[...].m__ files are used to generate the dataset
 * __BLDC_simulator_test[...].m__ were used to tune the nominal parameters of the simulator.

In *matlab_simulator/Model_id_scripts* and *matlab_simulator/EKF_scripts* the code for generating and tuning the EKFs for the different motor configurations can be found.

## Training

In the *speed_estimator* folder:
* __transformer_zerostep.py__ contains the architecture for the meta model
* __dataset.py__ is used to extract data batches from the dataset
* __train_zerostep.py__ is used to train the meta model
* __test_zerostep.py__ test the metamodel against the dataset

The code was originally taken and modified from [sysid-transformers](https://github.com/forgi86/sysid-transformers)


# Software requirements
Simulation was performed on Matlab R2024a

Model training was performed on a Python 3.12 environment with:

 * numpy
 * scipy
 * matplotlib
 * python-control
 * pytorch (v2.7.0)
 * Cuda 12.6

These dependencies may be installed through the commands:

```
pip install numpy scipy matplotlib control
```

For more details on pytorch installation options (e.g. support for CUDA acceleration), please refer to the official [installation instructions](https://pytorch.org/get-started/locally/).

The following packages are also useful:

```
pip install jupyter # (optional, to run the test jupyter notebooks)
pip install wandb # (optional, for experiment logging & monitoring)
```

# Hardware requirements
While all the scripts should be able run on CPU, execution may be frustratingly slow. For faster training, a GPU is highly recommended.
To run the paper's examples, we used a pc equipped with an nVidia RTX 3070ti GPU.




# Citing

If you find this project useful, we encourage you to cite the [paper](https://arxiv.org/abs/2504.00673) 
```
@article{colombo2025context,
  title={In-Context Learning for Zero-Shot Speed Estimation of BLDC motors},
  author={Colombo, Alessandro and Busetto, Riccardo and Breschi, Valentina and Forgione, Marco and Piga, Dario and Formentin, Simone},
  journal={arXiv preprint arXiv:2504.00673},
  year={2025}
}
```

# Look at:
- speed_estimator/transformer_zerostep.py - architettura Transformer (decoder-only) per stima @k dal contesto. 
- speed_estimator/train_zerostep.py - training loop; costruisce il batch, gestisce l'autoregressione (ri- . iniezione di @k-1 come 5° canale), loss, scheduler. 
- speed_estimator/dataset.py - normalizzazioni e finestre temporali (contesto lungo H); prepara (u, y) per il modello.
- speed_estimator/torch_utils.py - utility (device/GPU, ecc.). 
- speed_estimator/deprecated/KalmanFilter.py - baseline (E)KF per confronto con ICL (richiamo allo stato dell'arte).


# ⚙️ CUDA, PyTorch & GPU Compatibility

This appendix explains **why CUDA matters**, how to **verify your setup**, what **compute capability / `sm_XXX`** means, and how to **install the right PyTorch build** for your GPU. Includes quick troubleshooting and perf tips.

---

# Use a virtual environment (Windows / venv)

**Why it’s useful (evidence-based benefits):**
- **Isolation:** dependencies for this project don’t leak into (or break) other projects.
- **Reproducibility:** `requirements.txt` + a venv gives you a known, repeatable software stack.
- **Cleaner CUDA/PyTorch installs:** prevents mixing CPU wheels and CUDA wheels from a global Python.
- **No admin rights needed:** everything lives in the project folder.

## Create once (in repo root)

```powershell
# Ensure Python is on PATH
python --version

# Create .venv folder in the current directory
python -m venv .venv


---

## Why CUDA matters

To run training/inference on the **GPU**, three layers must align:

1. **GPU hardware** — e.g., RTX 5070 Ti Laptop = *Blackwell* → compute capability **12.0** = `sm_120`.
2. **NVIDIA driver** — `nvidia-smi` shows the *max* CUDA the driver supports (e.g., `CUDA Version: 12.9`).
3. **PyTorch wheel** — built for a specific CUDA toolkit **and** a set of GPU architectures.  
   If your wheel doesn’t include your `sm_XXX`, you’ll see warnings like `supports sm_50 ... sm_90` and runtime errors like `no kernel image is available`.

**TL;DR:** Driver’s CUDA and wheel’s CUDA **don’t have to match exactly** (driver must be ≥ wheel).  
What **must** match is **architecture support** (e.g., wheel includes `sm_120` kernels or PTX).

---

## Quick compatibility check

~~~bash
python - << 'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
x = torch.randn(256,256, device="cuda"); y = x @ x.T
print("OK:", y.is_cuda)
PY
~~~

**Interpretation**
- Capability `(12, 0)` + `OK: True` → kernels are running on your GPU.
- Warning `supports sm_50..sm_90` or error `no kernel image` → install a wheel that includes **sm_120**.
- `torch: ...+cpu` and `cuda: None` → you installed a **CPU-only** wheel.

Tip: where is PyTorch coming from?
~~~python
import torch; print(torch.__file__)
~~~
If it’s in `site-packages`, you’re using a wheel (good). If it’s inside a local repo, you’re on a source build.

---

## What is `sm_XXX` (compute capability)?

`sm_XXX` identifies the GPU micro-architecture:

| Architecture | Generation | Compute Capability |
|---|---|---|
| Pascal       | GTX 10xx   | `sm_60`, `sm_61`   |
| Volta        | V100       | `sm_70`            |
| Turing       | RTX 20xx   | `sm_75`            |
| Ampere       | A100/RTX30 | `sm_80`, `sm_86`   |
| Ada          | RTX 40xx   | `sm_89`            |
| Hopper       | H100       | `sm_90`            |
| Blackwell    | RTX 50xx   | **`sm_120`**       |

Your PyTorch build must include kernels (or PTX) for your `sm_XXX`.

---

## Install a compatible PyTorch (Windows / pip)

**Recommended (nightly channel for newest GPUs):** put these at the **top** of `requirements.txt`:

~~~text
--index-url https://download.pytorch.org/whl/nightly/cu129
--extra-index-url https://pypi.org/simple
--pre

torch
torchvision
torchaudio
~~~

Then install into your virtual environment:

~~~powershell
# inside your venv
python -m pip install -U pip
pip uninstall -y torch torchvision torchaudio
pip install -r requirements.txt --upgrade --no-cache-dir
~~~

**One-liner (no requirements file):**
~~~powershell
pip uninstall -y torch torchvision torchaudio
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu129 `
    torch torchvision torchaudio --no-cache-dir
~~~

If you prefer stability, use the latest **stable** `cu12x` that **includes your arch**. For brand-new GPUs, nightlies usually add support earlier.

> ⚠️ Avoid pinning exact dev builds like `==2.x.y.devYYYYMMDD+cu129` — nightlies get pruned. Prefer the channel + `--pre`.

---

## Common errors & quick fixes

- **`Torch not compiled with CUDA enabled`**  
  You installed a **CPU wheel**. Reinstall from the CUDA index (see above).

- **`supports sm_50 ... sm_90`** + **`no kernel image`**  
  Your wheel lacks your GPU arch (e.g., `sm_120`). Install a newer CUDA wheel (nightly cu129) or build from source with:
  ~~~powershell
  set TORCH_CUDA_ARCH_LIST=12.0
  ~~~
  (or `12.0+PTX` for forward-compat).

- **`nvidia-smi` shows 12.9, `torch.version.cuda` shows 12.6`**  
  Normal. Driver’s max CUDA ≠ wheel’s toolkit. Architecture support is the real blocker.

---

## Optional performance tips

Enable fast paths near startup:
~~~python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
~~~

Use mixed precision (faster & less memory):
~~~python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for x, y in train_dl:
    x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with autocast(dtype=torch.bfloat16):  # or torch.float16
        yhat = model(x)
        loss = criterion(yhat, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
~~~

DataLoader tuning:
~~~python
from torch.utils.data import DataLoader
loader = DataLoader(dataset,
                    batch_size=..., shuffle=True,
                    num_workers=4, pin_memory=True,
                    persistent_workers=True)
~~~

---

## Quick GPU micro-benchmark (optional)

~~~python
import torch, time, math
torch.backends.cuda.matmul.allow_tf32 = True

def time_gpu(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        t0=torch.cuda.Event(True); t1=torch.cuda.Event(True)
        t0.record(); fn(); t1.record(); torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1))
    ts.sort(); return sum(ts[len(ts)//2-1:len(ts)//2+1])/2

def gflops(M,N,K,dtype):
    a=torch.randn(M,K,device='cuda',dtype=dtype)
    b=torch.randn(K,N,device='cuda',dtype=dtype)
    ms=time_gpu(lambda: a@b)
    flops=2*M*N*K
    return flops/(ms/1e3)/1e12, ms

print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))
for dt,name in [(torch.float32,'fp32/tf32'),(torch.float16,'fp16'),(torch.bfloat16,'bf16')]:
    tflops, ms = gflops(4096,4096,4096, dt)
    print(f'GEMM {name}: {tflops:.2f} TFLOP/s  ({ms:.2f} ms)')
~~~

---

## FAQ

**Q: Why does `nvidia-smi` say CUDA 12.9 but `torch.version.cuda` say 12.6?**  
A: `nvidia-smi` shows the **driver’s** max CUDA. `torch.version.cuda` is the **wheel’s** toolkit. They don’t need to match; the driver must be ≥ wheel’s CUDA.

**Q: What is the minimum to remember?**  
A: Install a CUDA wheel that includes your **`sm_XXX`**, run the quick check snippet, and make sure `OK: True`.
