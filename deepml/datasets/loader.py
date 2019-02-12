
from PIL import Image
from torch.utils.data import Dataset


class DeepMLDataLoader(Dataset):
    """Data loader for deep metric learning

    Args:
        df_data (Dataframe): A dataframe contains two columns
            img: the path of each image
            label: the labels
    """

    def __init__(self, df_data, transform=None):
        super(DeepMLDataLoader, self).__init__()
        self.df_data = df_data
        self.transform = transform
        self.is_test = 'label' in df_data.columns

    def __getitem__(self, index):
        img_path = self.df_data['img'][index]
        img = Image.open(img_path).convert('RGB')
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        label = self.df_data['label'][index] if self.is_test else -1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df_data)
