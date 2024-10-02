from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SweepConfig:
    delta: float  # Percentage delta for sweeping
    steps: int  # Number of steps for sweeping

@dataclass
class BaseOptimizerConfig:
    lr: float
    weight_decay: float
    lr_sweep: Optional[SweepConfig] = None
    weight_decay_sweep: Optional[SweepConfig] = None

    def __post_init__(self):
        assert self.lr > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"

@dataclass
class SGDConfig(BaseOptimizerConfig):
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    momentum_sweep: Optional[SweepConfig] = None
    dampening_sweep: Optional[SweepConfig] = None

@dataclass
class AdamWConfig(BaseOptimizerConfig):
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    betas_sweep: Optional[SweepConfig] = None
    eps_sweep: Optional[SweepConfig] = None

# Add more optimizer configs as needed
