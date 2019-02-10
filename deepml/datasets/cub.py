import glob
import os

import pandas as pd

from .loader import DeepMLDataLoader


class Cub():
    def __init__(self, data_path):
        super(Cub, self).__init__()
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
        self.df_train = df[df['label'] <= 100].reset_index()
        self.df_test = df[df['label'] > 100].reset_index()

    def get_train_loader(self, transform):
        return DeepMLDataLoader(self.df_train, transform)

    def get_test_loader(self, transform):
        return DeepMLDataLoader(self.df_test, transform)
