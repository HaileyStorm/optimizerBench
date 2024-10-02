from training.base_trainer import BaseTrainer


class Domain3Trainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, device):
        super(Domain3Trainer, self).__init__(model, optimizer, criterion, device)

    def train_epoch(self, dataloader):
        # TODO: Implement training for one epoch
        pass

    def validate(self, dataloader):
        # TODO: Implement validation
        pass

    def train(self, train_dataloader, val_dataloader, num_epochs):
        # TODO: Implement full training loop
        # TODO: See note in main.py about training by step etc.
        pass
