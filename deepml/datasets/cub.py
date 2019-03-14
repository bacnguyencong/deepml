import glob
import os

import pandas as pd

from .dataset import Dataset


class Cub(Dataset):
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
        # create a map
        data_df = {
            'train': df[df['label'] <= 100].reset_index(drop=True),
            'test': df[df['label'] > 100].reset_index(drop=True)
        }
        # check if data were loaded correctly
        assert len(data_df['train']) == 5864 and len(data_df['test']) == 5924
        return data_df
