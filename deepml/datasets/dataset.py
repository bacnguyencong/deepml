from abc import ABC

from .loader import DeepMLDataset


class Dataset(ABC):
    def __init__(self, data_path):
        self.data_df = self.compute_dataframe(data_path)

    def get_dataset(self, ttype, transform, inverted):
        """Compute a dataset.

        Args:
            ttype (str): The type of dataset ('train', 'test', 'valid')
            transform ([type]): [description]

        Returns:
            Dataloader: The desired Dataset.
        """
        return DeepMLDataset(self.data_df[ttype], inverted, transform)

    def compute_dataframe(self, data_path):
        """Compute train, valid, and test dataframes

        Args:
            data_path (str): The path to data.

        Returns:
            map(str, pd.DataFrame): 'train', 'test', 'valid' dataframes.
        """
        raise NotImplementedError
