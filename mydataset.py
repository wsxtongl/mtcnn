import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
class Dataset(Dataset):
    def __init__(self,path):

        self.path = path
        self.data = []
        self.data.extend(open(os.path.join(path,"positive_txt")).readlines())
        self.data.extend(open(os.path.join(path, "negative_txt")).readlines())
        self.data.extend(open(os.path.join(path, "part_txt")).readlines())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        strs =self.data[index].strip().split()
        confidence = torch.Tensor([int(strs[1])])

        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        img = Image.open(os.path.join(self.path,strs[0]))
        tran = transforms.ToTensor()
        img = tran(img)
        return img,confidence,offset


if __name__ == '__main__':

    path = r"D:\mtcnn_sample\48" # 只以尺寸为48的为例
    dataset = Dataset(path)
    print(dataset[0])
