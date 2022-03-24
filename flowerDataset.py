import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class FlowersDataset(Dataset):
    def __init__(self, csv, root_dir, transform=None):
        self.labels = pd.read_csv(csv)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        imgpath = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        image = io.imread(imgpath)
        y_label = torch.tensor(int(self.labels.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
