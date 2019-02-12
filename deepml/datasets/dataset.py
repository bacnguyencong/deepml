from abc import ABC


class Dataset(ABC):
    def __init__(self, data_path):
        self.data_df = self.compute_dataframe(data_path)

    def get_dataloader(self, ttype, transform):
        """Compute a dataloader.

        Args:
            ttype (str): The type of dataloader ('train', 'test', 'valid')
            transform ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """

        raise NotImplementedError

    def compute_dataframe(self, data_path):
        """Compute train, valid, and test dataframes

        Args:
            data_path (str): The path to data.

        Returns:
            map(str, pd.DataFrame): 'train', 'test', 'valid' dataframes.
        """
        raise NotImplementedError
