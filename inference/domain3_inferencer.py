from inference.base_inferencer import BaseInferencer


class Domain3Inferencer(BaseInferencer):
    def __init__(self, model, device):
        super(Domain3Inferencer, self).__init__(model, device)

    def inference(self, dataloader):
        # TODO: Implement inference
        pass

    def evaluate(self, dataloader):
        # TODO: Implement evaluation
        pass
