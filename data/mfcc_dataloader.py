import torch
import os
from torch.utils.data import Dataset
import random
import numpy as np



class MFCC_dataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, path=None):
        # TODO
        # 1. Initialize file path or list of file names.
        self.path = path
        self.classes = os.listdir(self.path)
        self.mfcc2gender = []
        self.gender2id = {'f':0, 'm':1}
        for each_class in self.classes:
            uIDs = os.listdir(os.path.join(path,each_class))
            for uID in uIDs:
                uID_path = os.path.join(path,each_class,uID)
                self.mfcc2gender.append((uID_path,each_class))
        random.shuffle(self.mfcc2gender)
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        mfcc_file, gender = self.mfcc2gender[index]
        return torch.from_numpy(np.load(mfcc_file)[:59995]), torch.Tensor([self.gender2id[gender]]).long()
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.mfcc2gender)
