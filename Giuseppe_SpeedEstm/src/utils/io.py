import os, datetime

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def make_run_dir(root: str, exp_name: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = os.path.join(root, f"{ts}_{exp_name}")
    safe_mkdir(run)
    return run


# CHECKED -- all good!