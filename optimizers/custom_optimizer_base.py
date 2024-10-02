from torch.optim import Optimizer
from config.optimizer_config import BaseOptimizerConfig

class CustomOptimizerBase(Optimizer):
    def __init__(self, params, config: BaseOptimizerConfig):
        defaults = {
            'lr': config.lr,
            'weight_decay': config.weight_decay,
        }
        super(CustomOptimizerBase, self).__init__(params, defaults)

    def step(self, closure=None):
        raise NotImplementedError("Subclasses must implement step method")

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
