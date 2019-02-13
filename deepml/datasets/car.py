import os
import scipy.io
import pandas as pd

from .dataset import Dataset


class Car(Dataset):
    def compute_dataframe(self, data_path):
        # list of image labels and their paths
        mat = scipy.io.loadmat(os.path.join(data_path, 'cars_annos.mat'))
        img_list, labels = [], []
        # read all images and their labels
        for x in mat['annotations'][0]:
            img_list.append(os.path.join(data_path, x[0][0]))
            labels.append(x[5][0][0])
        df = pd.DataFrame({'img': img_list, 'label': labels})
        # create a map
        data_df = {
            'train': df[df['label'] <= 98].reset_index(),
            'valid': df[df['label'] <= 98].reset_index(),
            'test': df[df['label'] > 98].reset_index()
        }
        return data_df
