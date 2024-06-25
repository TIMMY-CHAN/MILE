import os
import torch
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
import matplotlib as plt
import matplotlib.image as mpig

import json
from scipy import ndimage
from scipy.ndimage import zoom
from PIL import Image
from torchvision import transforms
from nltk.corpus import wordnet 

class Slake_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        
        with open(label_dir, 'r', encoding='utf-8') as f:
            self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)
        if self.transform:
            data = self.transform(data)
            data = data/255

        question = sample["question"]
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid

class VQA_Instruction_dataset(Data.Dataset): # Dataloader for Instruction-format dataset
    def __init__(self, data_dir, label_dir, transform):    
        with open(label_dir, 'r', encoding='utf-8') as f:  
             self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]    
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)
        if self.transform:
            data = self.transform(data)
            data = data/255
        if list(data.size())[0] == 1:
            data = data.repeat(3, 1, 1)
        
        # Randomly select a question from question_pool in instruction_data
        question_pool = sample["question_pool"]
        question = random.choice(question_pool)

        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid