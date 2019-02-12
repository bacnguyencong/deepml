import glob
import os

import pandas as pd

from .dataset import Dataset
from .loader import DeepMLDataLoader


class Cub(Dataset):
    def __init__(self, data_path):
        super(Cub, self).__init__(data_path)

    def compute_dataframe(self, data_path):
        # path to all images
        img_path = os.path.join(data_path, 'CUB_200_2011/images')
        # list of image labels and their paths
        img_list = []
        for cur_dir in os.listdir(img_path):
            parent = os.path.join(img_path, cur_dir)
            img_id = int(cur_dir[0:3])
            for img in glob.glob(parent + '/*.jpg'):
                path = os.path.join(parent, img)
                img_list.append([path, img_id])
        df = pd.DataFrame(img_list, columns=['img', 'label'])

        df_data = dict()
        df_data['train'] = df[df['label'] <= 100].reset_index()
        df_data['valid'] = df[df['label'] <= 100].reset_index()
        df_data['test'] = df[df['label'] > 100].reset_index()

        return df_data

    def get_dataloader(self, ttype, transform):
        return DeepMLDataLoader(self.df_data[ttype], transform)
