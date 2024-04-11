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

class VQA_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r') as f:  
            for line in f:  
                self.sample_list = [json.loads(line) for line in f] 
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
    

class Slake_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r') as f:  
            #for line in f:  
            s = f.read() 
        self.sample_list = json.loads(s)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]      # 按值索引到图片名称
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


class rsna_vqa_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r') as f:  
            #for line in f:  
            s = f.read() 
        self.sample_list = json.loads(s)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        for key,value in sample.items():
            slice_name = key
            random_question = random.choice(list(value.keys()))
            item = random.choice(value[random_question])
        question = item['input']
        answer = item['output']   
        data_path = os.path.join(self.data_dir, slice_name+'.png')
        data = Image.open(data_path)   # 找到图片对应的png文件
        if self.transform:
            data = self.transform(data)
            data = data.repeat(3,1,1)
            data = data/255
        #print('type of data',type(data))
        answer = str(answer)
        return data, question, answer


class VQA_Instruction_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r', encoding='utf-8') as f:  
             self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
        if self.transform:
            data = self.transform(data)
            data = data/255

        # 将灰度图像通道数改为3，统一所有图像
        if list(data.size())[0] == 1:
            data = data.repeat(3, 1, 1)
        # print('new_data.size:', list(data.size()))
        # print('type of data',type(data))
        
        # 从instruction的opts池中，随机选择一个instruction作为question
        question_pool = sample["question_pool"]
        question = random.choice(question_pool)

        # 真实答案answer，即ground-truth
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid


class VQA_ordinary_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r', encoding='utf-8') as f:  
             self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
        
        if self.transform:
            data = self.transform(data)
            data = data/255
        if list(data.size())[0] == 1:
            data = data.repeat(3, 1, 1)
        # print('new_data.size:', list(data.size()))
        # print('type of data',type(data))
        
        # 非instruction数据集，只有一个question
        question = sample["question"]

        # 真实答案answer，即ground-truth
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid


class VQA_slake_test_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r', encoding='utf-8') as f:  
            self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        old_img_name = sample["img_name"] 
        slice_name = old_img_name.replace('xmlab', 'slake_')
        slice_name = slice_name.replace('/source.jpg', '.png')  # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        # print('img_path: ', data_path)
        data = Image.open(data_path)   # 找到图片对应的png文件
        if self.transform:
            data = self.transform(data)
            data = data/255
        if list(data.size())[0] == 1:
            data = data.repeat(3, 1, 1)
        # print('new_data.size:', list(data.size()))
        # print('type of data',type(data))
        
        # 非instruction数据集，只有一个question
        question = sample["question"]

        # 真实答案answer，即ground-truth
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid

class slake_gauss_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r', encoding='utf-8') as f:  
             self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)
    
    def add_gaussian_noise(self, img):
        img = np.array(img)
        mean = 0
        var = 0.1  # 这里可以调整高斯噪声的方差
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, img.shape)
        noisy_img = img + gauss
        noisy_img = np.clip(noisy_img, 0, 255)  # 将像素值限制在0到255之间
        noisy_img = noisy_img.astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    
    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
    
        # 以0.5的概率随机添加高斯噪声
        if random.random() < 0.5:
            data = self.add_gaussian_noise(data)
    
        if self.transform:
            data = self.transform(data)
            data = data/255
        if list(data.size())[0] == 1:
            data = data.repeat(3, 1, 1)
        # print('new_data.size:', list(data.size()))
        # print('type of data',type(data))
        
        # 非instruction数据集，只有一个question
        question = sample["question"]

        # 真实答案answer，即ground-truth
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid

class slake_TextSwap_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r', encoding='utf-8') as f:  
             self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)
    
########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words
    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words
    
    
    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
       
        if self.transform:
            data = self.transform(data)
            data = data/255
        if list(data.size())[0] == 1:
            data = data.repeat(3, 1, 1)
        # print('new_data.size:', list(data.size()))
        # print('type of data',type(data))
        
        # 非instruction数据集，只有一个question
        question = sample["question"]
        word_list = question.split() # Splitting the input text into a list of words
        # 随即交换两个词，2次
        question = " ".join(self.random_swap(word_list, 2))
         
        # 真实答案answer，即ground-truth
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid

class slake_TextDelete_dataset(Data.Dataset):
    def __init__(self, data_dir, label_dir, transform):        # data_dir和label_dir分别表示图像和标签路径
        with open(label_dir, 'r', encoding='utf-8') as f:  
             self.sample_list = json.load(f)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)
    
########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
    def random_deletion(words, p):
        #obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words
    
    
    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        slice_name = sample["img_name"]      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
       
        if self.transform:
            data = self.transform(data)
            data = data/255
        if list(data.size())[0] == 1:
            data = data.repeat(3, 1, 1)
        # print('new_data.size:', list(data.size()))
        # print('type of data',type(data))
        
        # 非instruction数据集，只有一个question
        question = sample["question"]
        word_list = question.split() # Splitting the input text into a list of words
        question = " ".join(self.random_deletion(word_list, 0.2))
        
        
        # 真实答案answer，即ground-truth
        answer = sample['answer']
        answer = str(answer)
        qid = sample['qid']
        return data, question, answer, qid