1. Project Plan:

The ML optimizer test bench will consist of the following main components:
a. Models: Implementations of 2-3 domain-specific models
b. Datasets: Data loaders for each domain/task
c. Optimizers: Built-in and custom optimizer implementations
d. Training: Modules for training models with different optimizers
e. Inference: Modules for model inference and evaluation
f. Logging: Wandb integration for result logging and analysis
g. Configuration: Classes for managing model and optimizer parameters
h. Main Script: Coordinates the test bench execution

These components will interact as follows:
1. The main script loads configurations and initializes models, datasets, and optimizers.
2. For each domain/task and optimizer combination:
   a. The training module trains the model using the specified optimizer.
   b. The inference module evaluates the model's performance.
   c. The logging module records results to wandb.
3. Results are organized for easy comparison across optimizers and domains.

2. File/Folder Structure:

The "domain#" prefix in many of the files below will be renamed according to selected domains/tests (but have been created with these names for now). See DOMAIN_CONFIG dictionary below.

```
ml_optimizer_testbench/
│
├── main.py
├── config/
│   ├── __init__.py
│   ├── model_config.py
│   └── optimizer_config.py
│
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── image_classification_model.py
│   ├── domain2_model.py
│   └── domain3_model.py
│
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── image_classification_dataset.py
│   ├── domain2_dataset.py
│   └── domain3_dataset.py
│
├── optimizers/
│   ├── __init__.py
│   ├── custom_optimizer_base.py
│   └── custom_optimizer1.py
│
├── training/
│   ├── __init__.py
│   ├── base_trainer.py
│   ├── image_classification_trainer.py
│   ├── domain2_trainer.py
│   └── domain3_trainer.py
│
├── inference/
│   ├── __init__.py
│   ├── base_inferencer.py
│   ├── image_classification_inferencer.py
│   ├── domain2_inferencer.py
│   └── domain3_inferencer.py
│
└── utils/
    ├── __init__.py
    ├── logging.py
    └── helpers.py
```

3. Key Classes and Relationships:

a. BaseModel (models/base_model.py):
   - Inherits from nn.Module
   - Provides common functionality for all models

b. ImageClassificationModel, Domain2Model, Domain3Model (models/domain*_model.py):
   - Inherit from BaseModel
   - Implement specific architectures for each domain

c. BaseDataset (datasets/base_dataset.py):
   - Inherits from torch.utils.data.Dataset
   - Provides common functionality for all datasets

d. ImageClassificationDataset, Domain2Dataset, Domain3Dataset (datasets/domain*_dataset.py):
   - Inherit from BaseDataset
   - Implement specific data loading and preprocessing for each domain

e. CustomOptimizerBase (optimizers/custom_optimizer_base.py):
   - Inherits from torch.optim.Optimizer
   - Provides a base class for custom optimizers

f. CustomOptimizer1 (optimizers/custom_optimizer1.py):
   - Inherits from CustomOptimizerBase
   - Implements a custom optimizer

g. BaseTrainer (training/base_trainer.py):
   - Provides common training functionality

h. ImageClassificationTrainer, Domain2Trainer, Domain3Trainer (training/domain*_trainer.py):
   - Inherit from BaseTrainer
   - Implement domain-specific training logic

i. BaseInferencer (inference/base_inferencer.py):
   - Provides common inference functionality

j. ImageClassificationInferencer, Domain2Inferencer, Domain3Inferencer (inference/domain*_inferencer.py):
   - Inherit from BaseInferencer
   - Implement domain-specific inference logic

k. ModelConfig (config/model_config.py):
   - Manages model parameters and configurations

l. OptimizerConfig (config/optimizer_config.py):
   - Manages optimizer parameters, hyperparameters, and sweep configurations

m. WandbLogger (utils/logging.py):
   - Handles logging to wandb

4. Main Script Structure (main.py):

The main script will coordinate the test bench execution:

1. Import necessary modules and classes
2. Define configuration loading functions
3. Import the appropriate (to be renamed) files and define a domain configuration dictionary:
   ```python
   DOMAIN_CONFIG = {
       'imageClassification': {
           'model': ImageClassificationModel,
           'model_config': ImageClassificationModelConfig,
           'dataset': ImageClassificationDataset,
           'trainer': ImageClassificationTrainer,
           'inferencer': ImageClassificationInferencer
       },
       'domain2': {
           ...
       },
       # Add more domains as needed
   }
   ```
4. Define optimizer initialization function
5. Define model initialization function
6. Define main execution loop:
   a. Load configurations
   b. Initialize models, datasets, and optimizers
   c. For each domain/task:
      - For each optimizer:
        - Initialize model and optimizer
        - Create trainer and inferencer instances
        - Run training loop
        - Run inference/evaluation
        - Log results to wandb
   d. Organize and summarize results

This structure provides a modular and extensible framework for testing various optimizers against different models and domains, adhering to the specified constraints and requirements.

---

The following has been completed:

1. config/
   - model_config.py
     - BaseModelConfig (dataclass, implemented)
   - optimizer_config.py
     - SweepConfig (dataclass, implemented)
     - BaseOptimizerConfig (dataclass, implemented)
     - SGDConfig (dataclass, implemented - will be removed/replaced when custom optimizer(s) defined)
     - AdamWConfig (dataclass, implemented)

2. models/
   - base_model.py
     - BaseModel(nn.Module) (implemented)
       - __init__(self, config: BaseModelConfig)
       - _build_layers(self) [abstract]
       - forward(self, x) [abstract]
       - get_num_parameters(self)
       - reset_parameters(self)
   - image_classification_model.py
     - ImageClassificationModelConfig(BaseModelConfig) (implemented)
     - ImageClassificationModel(BaseModel) (implemented)
   - domain2_model.py, domain3_model.py
     - Domain[2/3]ModelConfig(BaseModelConfig) (skeleton)
     - Domain[2/3]Model(BaseModel) (skeleton)

3. datasets/
   - base_dataset.py
     - BaseDataset(Dataset) (implemented)
       - __init__(self)
       - __len__(self) [abstract]
       - __getitem__(self, idx) [abstract]
       - preprocess(self, data) [abstract]
   - image_classification_dataset.py
     - ImageClassificationDataset(BaseDataset) (implemented)
     - get_image_classification_dataloaders() (implemented)
   - domain2_dataset.py, domain3_dataset.py
     - Domain[2/3]Dataset(BaseDataset) (skeleton)

4. optimizers/
   - custom_optimizer_base.py
     - CustomOptimizerBase(Optimizer) (implemented)
       - __init__(self, params, config: BaseOptimizerConfig)
       - step(self, closure=None) [abstract]
       - zero_grad(self)
   - custom_optimizer1.py
     - CustomOptimizer1(CustomOptimizerBase) (skeleton)

5. training/
   - base_trainer.py
     - BaseTrainer (implemented)
       - __init__(self, model, optimizer, criterion, device)
       - initialize_training(self)
       - setup_dataloaders(self) [abstract]
       - train_step(self)
       - validate(self)
       - save_checkpoint(self, path)
       - load_checkpoint(self, path)
   - image_classification_trainer.py
     - ImageClassificationTrainer(BaseTrainer) (implemented)
       - __init__(self, model, optimizer, device)
       - setup_dataloaders(self)
   - domain2_trainer.py, domain3_trainer.py
     - Domain[2/3]Trainer(BaseTrainer) (skeleton)

6. inference/
   - base_inferencer.py
     - BaseInferencer (implemented)
       - __init__(self, model, device)
       - inference(self, dataloader) [abstract]
       - evaluate(self, dataloader) [abstract]
   - image_classification_inferencer.py, domain2_inferencer.py, domain3_inferencer.py
     - Domain[1/2/3]Inferencer(BaseInferencer) (skeleton)

7. utils/
   - logging.py
     - WandbLogger (skeleton)
       - __init__(self, project_name, config)
       - log(self, data)
       - finish(self)

8. main.py
   - DOMAIN_CONFIG (dictionary, implemented)
   - load_model_config(domain) (implemented)
   - load_optimizer_config() (placeholder)
   - initialize_optimizer(optimizer_name, model_parameters, config) (implemented)
   - initialize_model(domain, config) (implemented)
   - generate_sweep_configs(base_config) (implemented)
   - main() (implemented)