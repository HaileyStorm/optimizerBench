class BaseInferencer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def inference(self, dataloader):
        raise NotImplementedError("Subclasses must implement inference method")

    def evaluate(self, dataloader):
        raise NotImplementedError("Subclasses must implement evaluate method")
