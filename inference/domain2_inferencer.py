from inference.base_inferencer import BaseInferencer


class Domain2Inferencer(BaseInferencer):
    def __init__(self, model, device):
        super(Domain2Inferencer, self).__init__(model, device)

    def inference(self, dataloader):
        # TODO: Implement inference
        pass

    def evaluate(self, dataloader):
        # TODO: Implement evaluation
        pass
