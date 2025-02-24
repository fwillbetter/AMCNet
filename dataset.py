from typing import List
import torch
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import json


class TxtDataset(Dataset):

    def __init__(self, file_path: str,
                     img_path: str,
                     ignore_labels: List[str],
                     mode='train',
                     transform=None):
        """
        初始化类的属性

        Args:
            file_path (str): 文件路径
            img_path (str): 图像路径
            ignore_labels (List[str]): 忽略的标签列表
            mode (str, optional): 模式，默认为'train '
            transform (callable, optional): 图像转换函数
        """
        with open(file_path, 'r') as file:
            self.img_path = img_path
            label_list = []
            image_list = []
            self.ignore_labels = ignore_labels

            if mode == 'train':
                for line in file:
                    label, image = line.split()
                    label = label.split()[-1]
                    image = image.split()[0][1:]
                    if int(label) not in ignore_labels:
                        label_list.append(label)
                        image_list.append(image)
            else:
                for line in file:
                    label, image = line.split()
                    label = label.split()[-1]
                    if int(label) not in ignore_labels:
                        label_list.append(label)
                        image_list.append(image)

            self.images = image_list
            self.labels = label_list
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """
        获取索引对应的图像和标签

        Args:
            item (int): 图像索引

        Returns:
            tuple: 包含图像和标签的元组
        """
        img_name = self.images[item]
        img_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_path)
        # img = Image.open(img_path)
        label = int(self.labels[item])
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        label = torch.tensor(label)
        # print(img_path)
        return img, label


class JsonDataset(Dataset):
    def __init__(self, file_path: str,
                 img_path: str,
                 transform=None):
        # 读取文件
        self.df = pd.read_json(file_path)
        # 传入图片路径
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_name = self.df['image_dict'][item]['img_path']
        img_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_path)
        img = Image.fromarray(np.uint8(img))
        # img = Image.open(img_path)
        label = int(self.df['image_dict'][item]['level_2'])
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        label = torch.tensor(label)
        return img, label


# 读取AI challenger数据集json
class AiJsonDataset(Dataset):

    def __init__(self, file_path: str,
                 img_path: str,
                 transform=None):
        # 读取文件
        # 原始文件文件格式是jsonlines而不是json，因此在读取的时候需要一条一条的读
        # self.df = pd.read_json(file_path)
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append((json.loads(line)))
        self.df = pd.DataFrame(data)
        # print(self.df.values[0][0]['image_id'])
        # 传入图片路径
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        # x = len(self.df.values)
        # y = len(self.df.T)
        # 返回数据集长度
        return len(self.df.T)
        # return len(self.df)  # 测试用，每轮训练一个样本

    def __getitem__(self, item):
        img_name = self.df.values[0][item]['image_id']
        img_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_path)
        # cv2.imshow("123", img)
        # cv2.waitKey(10)
        img = Image.fromarray(np.uint8(img))
        # img = Image.open(img_path)
        label = int(self.df.values[0][item]['label_id'])
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        label = torch.tensor(label)
        return img, label

# 读取BDD100k数据集json
class BDD100kJsonDataset(Dataset):

    def __init__(self, file_path: str,
                 img_path: str,
                 transform=None):
        # 初始化函数，用于读取文件和设置图片路径及转换函数
        # 参数：
        # - file_path: str，文件路径
        # - img_path: str，图片路径
        # - transform: 转换函数，默认为None

        # 读取文件
        # 原始文件文件格式是jsonlines而不是json，因此在读取的时候需要一条一条的读
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append((json.loads(line)))
        self.df = pd.DataFrame(data)
        # print(self.df.values[0][0]['image_id'])

        # 传入图片路径
        self.img_path = img_path

        # 设置转换函数
        self.transform = transform


    def __len__(self):
        # x = len(self.df)
        # y = len(self.df.T)
        # 返回数据集长度
        return len(self.df.T)

    def __getitem__(self, item):
        img_name = self.df.values[0][item]['name']
        img_path = os.path.join(self.img_path, img_name + '.jpg')
        img = cv2.imread(img_path)
        # cv2.imshow("123", img)
        # cv2.waitKey(10)
        img = Image.fromarray(np.uint8(img))
        # img = Image.open(img_path)
        label = int(self.df.values[0][item]['scene'])
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        label = torch.tensor(label)
        return img, label