import os
import torch
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
import matplotlib as plt
import matplotlib.image as mpig
#import cv2
import json
from scipy import ndimage
from scipy.ndimage import zoom
from PIL import Image
from torchvision import transforms


class VQA_Tuning_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r', encoding='utf-8') as f:  
             self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["image_name"]      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
        if self.transform:
            data = self.transform(data)
            data = data/255
        #print('type of data',type(data))
        question = sample["question"]
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid
    