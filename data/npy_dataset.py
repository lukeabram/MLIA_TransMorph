import numpy as np
import torch
from torch.utils.data import Dataset


import torch.nn.functional as F


        


class NPYBrainDataset(Dataset):
    def __init__(self, img_path, transforms=None):
        self.images = np.load(img_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx, 0]
        y = self.images[idx, 1]
        
        
        x = x[None, ...]
        y = y[None, ...]
        

        
        if self.transforms:
            x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        return x, y