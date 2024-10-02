import torch
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.current_epoch = 0
        self.train_step = 0
        self.best_val_metric = float('inf')  # Lower is better, change if needed

    def initialize_training(self):
        self.setup_dataloaders()
        self.train_iterator = iter(self.train_loader)

    def setup_dataloaders(self):
        raise NotImplementedError("Subclasses must implement setup_dataloaders method")

    def train_step(self):
        self.model.train()
        try:
            inputs, targets = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            inputs, targets = next(self.train_iterator)
            self.current_epoch += 1

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        return loss.item()

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'train_step': self.train_step,
            'best_val_metric': self.best_val_metric
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_step = checkpoint['train_step']
        self.best_val_metric = checkpoint['best_val_metric']
