from __future__ import annotations
import os, json, torch

def save_checkpoint(path: str, path_best: str, 
                    model, optimizer, model_args: dict, 
                    epoch: int, history: list[dict], 
                    cfg: dict, train_time: float, best: tuple = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "model_args": model_args,
        "epoch": epoch,
        "history": history,
        "cfg": cfg,
        "train_time": train_time,
    }, path)

    """if best:
        with open(os.path.join(os.path.dirname(path), "metrics_val.json"), "w", encoding="utf-8") as f:
            json.dump(history[-1], f, indent=2)"""


    if best is not None: 
        os.makedirs(os.path.dirname(path_best), exist_ok=True)
        best_state_dict, best_opt_dict, best_val_loss, best_epoch, best_history, best_time = best
        torch.save(
            {
                "model": best_state_dict,                          # BEST weights
                "optimizer": best_opt_dict,
                "best_epoch": best_epoch,
                "time": best_time,
                "best_val_loss": best_val_loss,
                "history": best_history,                               # train / val curves
                "cfg": cfg,
            },
            path_best,
        )


# CHECKED -- all good!