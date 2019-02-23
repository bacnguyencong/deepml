import os

import pandas as pd

from .dataset import Dataset


class Stanford(Dataset):
    """This is the Online Products dataset in [1] for metric learning.
    These images are collected from eBay using its provided
    API: https://go.developer.ebay.com/what-ebay-api

    [1] Oh Song, H., Xiang, Y., Jegelka, S. and Savarese, S., 2016. Deep
        metric learning via lifted structured feature embedding. In
        Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition (pp. 4004-4012).
    """

    def compute_dataframe(self, data_path):
        # path to all images
        root = os.path.join(data_path, 'Stanford_Online_Products')
        # create a map
        data_df = {
            'train': self.compute(root, 'Ebay_train.txt'),
            'valid': self.compute(root, 'Ebay_train.txt'),
            'test': self.compute(root, 'Ebay_test.txt')
        }
        # check if data were loaded correctly
        assert len(data_df['train']) == 59551 and len(data_df['test']) == 60502
        return data_df

    def compute(self, root_path, filename):
        df = pd.read_csv(os.path.join(root_path, filename), sep=' ')
        df = df.drop(columns=['image_id', 'super_class_id'])
        df['img'] = df['path'].map(lambda x: os.path.join(root_path, x))
        df['label'] = df['class_id']
        return df[['img', 'label']]
