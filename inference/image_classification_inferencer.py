from inference.base_inferencer import BaseInferencer


class ImageClassificationInferencer(BaseInferencer):
    def __init__(self, model, device):
        super(ImageClassificationInferencer, self).__init__(model, device)

    def inference(self, dataloader):
        # TODO: Implement inference
        pass

    def evaluate(self, dataloader):
        # TODO: Implement evaluation
        pass
