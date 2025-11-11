# src package marker; re-export common entry points for convenience
from .datasets import build_dataset
from .models import build_model
from .losses import build_loss
from .optim.factory import build_optimizer_and_scheduler
