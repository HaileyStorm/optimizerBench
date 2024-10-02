import copy
import torch
from config.model_config import BaseModelConfig
from config.optimizer_config import BaseOptimizerConfig, SweepConfig
from utils.logging import WandbLogger

# Import domain-specific classes
from models.image_classification_model import ImageClassificationModel, ImageClassificationModelConfig
from datasets.image_classification_dataset import ImageClassificationDataset
from training.image_classification_trainer import ImageClassificationTrainer
from inference.image_classification_inferencer import ImageClassificationInferencer

# TODO: Import other domain-specific classes as needed

DOMAIN_CONFIG = {
    'imageClassification': {
        'model': ImageClassificationModel,
        'model_config': ImageClassificationModelConfig,
        'dataset': ImageClassificationDataset,
        'trainer': ImageClassificationTrainer,
        'inferencer': ImageClassificationInferencer
    },
    # TODO: Add more domains as needed
}

def load_model_config(domain):
    if domain == 'imageClassification':
        return ImageClassificationModelConfig(num_classes=10, input_channels=3)
    # TODO: Add configurations for other domains
    raise ValueError(f"Unknown domain: {domain}")

def load_optimizer_config():
    # This is a placeholder/simple example. Custom optimizers should implement a BaseOptimizerConfig, which defines sweeps & should be loaded here.
    return BaseOptimizerConfig(
        lr=SweepConfig(0.001, 0.1, 3),
        weight_decay=SweepConfig(1e-5, 1e-3, 3)
    )

def initialize_optimizer(optimizer_name, model_parameters, config):
    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(model_parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model_parameters, lr=config.lr, weight_decay=config.weight_decay)
    # TODO: Add more optimizers as needed
    raise ValueError(f"Unknown optimizer: {optimizer_name}")

def initialize_model(domain, config):
    model_class = DOMAIN_CONFIG[domain]['model']
    return model_class(config)

def generate_sweep_configs(base_config):
    sweep_configs = [base_config]  # Start with the base configuration

    for param, sweep in vars(base_config).items():
        if isinstance(sweep, SweepConfig):
            new_configs = []
            for config in sweep_configs:
                base_value = getattr(config, param)
                for step in range(sweep.steps):
                    delta = base_value * sweep.delta * (step - (sweep.steps - 1) / 2) / ((sweep.steps - 1) / 2)
                    new_config = copy.deepcopy(config)
                    setattr(new_config, param, base_value + delta)
                    new_configs.append(new_config)
            sweep_configs = new_configs

    return sweep_configs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = "optimizer_testbench"
    total_steps = 10000  # Adjust as needed
    validation_interval = 100  # Adjust as needed
    optimizers = ['AdamW', 'SGD']  # Add more optimizers as needed (including custom optimizers when defined)

    # Some pseudo/skeleton/partial code:
    for domain, domain_config in DOMAIN_CONFIG.items():
        model_config = load_model_config(domain)
        base_optimizer_config = load_optimizer_config()  # TODO: this needs to work per-optimizer, see above

        for optimizer_name in optimizers:
            # Generate sweep configurations
            sweep_configs = generate_sweep_configs(base_optimizer_config)

            for sweep_config in sweep_configs:
                model = initialize_model(domain, model_config)
                optimizer = initialize_optimizer(optimizer_name, model.parameters(), sweep_config)

                trainer = domain_config['trainer'](model, optimizer, device)
                trainer.initialize_training()

                logger = WandbLogger(project_name, config={**model_config, **sweep_config})

                for step in range(total_steps):
                    loss = trainer.train_step()
                    logger.log({"train_loss": loss, "step": trainer.train_step, "epoch": trainer.current_epoch})

                    if step % validation_interval == 0:
                        val_loss, val_accuracy = trainer.validate()
                        logger.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "step": trainer.train_step,
                                    "epoch": trainer.current_epoch})

                        if val_loss < trainer.best_val_metric:
                            trainer.best_val_metric = val_loss
                            trainer.save_checkpoint(f"best_model_{domain}_{optimizer_name}.pth")

                # Final evaluation
                trainer.load_checkpoint(f"best_model_{domain}_{optimizer_name}.pth")
                inferencer = domain_config['inferencer'](trainer.model, device)
                test_loss, test_accuracy = inferencer.evaluate(trainer.test_loader)
                logger.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

                logger.finish()

if __name__ == "__main__":
    main()
