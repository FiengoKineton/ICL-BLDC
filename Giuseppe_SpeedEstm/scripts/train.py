from __future__ import annotations                                              ##
import argparse, os, yaml, torch                                                ##
from torch.utils.data import DataLoader                                         ##
from src.datatypes import load_yaml, merge_cfg, apply_overrides, dict_to_cfg    ## -- ##
from src.datasets import build_dataset, collate_batch                           ## -- ##
from src.models import build_model                                              ## -- ##
from src.optim.factory import build_optimizer_and_scheduler                     ## -- ##
from src.engine.seed import seed_everything                                     ## -- ##
from src.engine.trainer import train_loop                                       ## -- ##
from src.engine.checkpoint import save_checkpoint                               ## -- ##
from src.utils.io import make_run_dir                                           ## -- ##
from src.utils.plotting import plot_history                                     ## -- ##


# python -m scripts.train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml"])
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    cfg_dicts = [load_yaml(p) for p in args.config]
    cfg = merge_cfg(*cfg_dicts)
    cfg = apply_overrides(cfg, args.override)
    cfg_obj = dict_to_cfg(cfg)

    seed_everything(cfg_obj.seed)
    device = torch.device(cfg_obj.device if torch.cuda.is_available() or cfg_obj.device=="cpu" else "cpu")

    train_ds = build_dataset(cfg_obj.data.name, cfg["data"], "train")
    val_ds   = build_dataset(cfg_obj.data.name, cfg["data"], "val")
    train_loader = DataLoader(train_ds, batch_size=cfg_obj.data.batch_size, pin_memory=True, shuffle=True, num_workers=cfg_obj.data.num_workers, collate_fn=collate_batch) # base code stops at shuffle
    val_loader   = DataLoader(val_ds, batch_size=cfg_obj.data.batch_size, pin_memory=True, shuffle=False, num_workers=cfg_obj.data.num_workers, collate_fn=collate_batch)

    model, model_args = build_model(cfg_obj.model.name, cfg["model"])
    model = model.to(device)
    #if cfg_obj.model.compile: model = torch.compile(model)
    optimizer, scheduler_fn = build_optimizer_and_scheduler(model, device, IndentationError(cfg["train"]["epochs"]), cfg["optim"])

    print("check")
    history, best = train_loop( model=model, 
                                optimizer=optimizer, 
                                scheduler_fn=scheduler_fn, 
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                cfg=cfg, 
                                device=device, )

    run_dir = make_run_dir(args.runs, cfg_obj.exp_name)
    save_checkpoint(path=os.path.join(run_dir, "ckpt_last.pt"), 
                    path_best=os.path.join(run_dir, "ckpt_best.pt"),
                    model=model, 
                    model_args=model_args,
                    optimizer=optimizer, 
                    epoch=history[-1]["epoch"] if history else 0, 
                    history=history, 
                    cfg=cfg, 
                    best=best)
    plot_history(history, run_dir)
    with open(os.path.join(run_dir, "cfg.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved run in {run_dir}")
    print(f"Best validation loss: {best[1]}, at epoch: {best[2]}")

if __name__ == "__main__":
    main()
