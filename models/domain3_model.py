from dataclasses import dataclass
from config.model_config import BaseModelConfig
from models.base_model import BaseModel
import torch.nn as nn


@dataclass
class Domain3ModelConfig(BaseModelConfig):
    specific_param: float


class Domain3Model(BaseModel):
    def __init__(self, config: Domain3ModelConfig):
        super(Domain3Model, self).__init__(config)

    def _build_layers(self):
        layers = []
        in_features = self.config.input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(getattr(nn, self.config.activation)())
            layers.append(nn.Dropout(self.config.dropout_rate))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, self.config.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
