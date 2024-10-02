import torch.nn as nn
from .base_trainer import BaseTrainer
from datasets.image_classification_dataset import get_image_classification_dataloaders

class ImageClassificationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, device):
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, optimizer, criterion, device)

    def setup_dataloaders(self):
        self.train_loader, self.val_loader = get_image_classification_dataloaders()
        self.test_loader = self.val_loader  # Use val_loader for testing
