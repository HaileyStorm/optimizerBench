import torch
import torch.nn as nn
from .base_model import BaseModel
from config.model_config import BaseModelConfig

class ImageClassificationModelConfig(BaseModelConfig):
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

class ImageClassificationModel(BaseModel):
    def __init__(self, config: ImageClassificationModelConfig):
        super().__init__(config)

    def _build_layers(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.config.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.config.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x