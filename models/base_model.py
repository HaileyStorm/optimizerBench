import torch.nn as nn
from config.model_config import BaseModelConfig


class BaseModel(nn.Module):
    def __init__(self, config: BaseModelConfig):
        super(BaseModel, self).__init__()
        self.config = config
        self.layers = self._build_layers()

    def _build_layers(self):
        raise NotImplementedError("Subclasses must implement _build_layers method")

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
