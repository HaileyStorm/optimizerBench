from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__ method")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__ method")

    def preprocess(self, data):
        raise NotImplementedError("Subclasses must implement preprocess method")
