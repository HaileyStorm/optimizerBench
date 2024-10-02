from optimizers.custom_optimizer_base import CustomOptimizerBase
from config.optimizer_config import BaseOptimizerConfig

# TODO: Define CustomOptimizer1Config

class CustomOptimizer1(CustomOptimizerBase):
    def __init__(self, params, config: BaseOptimizerConfig):
        # TODO: Create defaults dictionary from CustomOptimzier1Config & pass appropriate config/handle super appropriately
        super(CustomOptimizer1, self).__init__(params, config)

    def step(self, closure=None):
        # TODO: Implement the optimization step
        pass