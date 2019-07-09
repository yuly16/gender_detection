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
        mfcc_np = np.load(mfcc_file)
        # mfcc_shape = mfcc_np.shape
        #if mfcc_shape[0] < 300:
        #    mfcc_np = np.pad(mfcc_np,((0,300-mfcc_shape[0]),(0,0)),'constant',constant_values = (0,0))
        #else:
        #    mfcc_np = mfcc_np[:300,:]
        return torch.from_numpy(mfcc_np).transpose(1,0), torch.Tensor([self.gender2id[gender]]).long()
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.mfcc2gender)
