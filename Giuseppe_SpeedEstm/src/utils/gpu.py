import torch
def pick_device(pref: str = "cuda"):
    return torch.device("cuda" if pref == "cuda" and torch.cuda.is_available() else "cpu")


# NOT USED!