import wandb


class WandbLogger:
    def __init__(self, project_name, config):
        self.run = wandb.init(project=project_name, config=config)

    def log(self, data):
        wandb.log(data)

    def finish(self):
        wandb.finish()
