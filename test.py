import os, wandb
print("ENV ENTITY =", os.environ.get("WANDB_ENTITY"))
print("ENV PROJECT=", os.environ.get("WANDB_PROJECT"))

run = wandb.init(
    entity="g7-fiengo",
    project="in-context-bldc",
    name="diag-run",
    finish_previous=True  # <-- qui, non dentro Settings
)

print("EFFECTIVE  ->", run.entity, run.project, run.url)
wandb.log({"ping": 1})
run.finish()
