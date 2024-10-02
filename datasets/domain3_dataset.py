from datasets.base_dataset import BaseDataset


class Domain3Dataset(BaseDataset):
    def __init__(self, data_path):
        super(Domain3Dataset, self).__init__()
        # TODO: Initialize dataset

    def __len__(self):
        # TODO: Return the length of the dataset
        pass

    def __getitem__(self, idx):
        # TODO: Return the item at the given index
        pass

    def preprocess(self, data):
        # TODO: Implement data preprocessing
        pass
