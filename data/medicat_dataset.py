import os
import torch
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
import matplotlib as plt
import matplotlib.image as mpig
#import cv2

from scipy import ndimage
from scipy.ndimage import zoom
from PIL import Image
from torchvision import transforms


class Medicat_dataset(Data.Dataset):
    def __init__(self, data_dir, label_csv, list_dir,transform):        # data_dir和label_dir分别表示图像和标签路径
        self.list_dir = list_dir    # 索引目录路径
        self.sample_list = open(list_dir).readlines()   # 采样图片名称
        self.data_dir = data_dir
        self.label_csv = label_csv
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
        if self.transform:
            data = self.transform(data)
            data = data/255
        label_path = slice_name     # 按标签索引.csv文件的某行
        caption = self.label_csv.loc[label_path, 'Caption']  # 找到图片对应标签
        return data, caption
    
