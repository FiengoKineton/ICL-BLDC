from __future__ import annotations
import argparse, os, json, torch, yaml
from torch.utils.data import DataLoader
from src.datatypes import load_yaml
from src.datasets import build_dataset, collate_batch
from src.models import build_model
from src.engine.evaluator import rollout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    device = torch.device(cfg.get("device", "cpu"))
    model = build_model(cfg["model"]["name"], cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    ds = build_dataset(cfg["data"]["name"], cfg["data"], args.split)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=collate_batch)
    outs = []
    for batch in loader:
        y_hat = rollout(model, batch, device)
        outs.append(y_hat.cpu())
    print(f"Evaluated {len(outs)} batches.")

if __name__ == "__main__":
    main()
