import os
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import cv2


class ImageFolderWithPaths(ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = ((path,) + original_tuple)
        return tuple_with_path


class SegmentationSet2(Dataset):
    def __init__(self,
                 images,
                 annotations,
                 transform=None):
        self.images = images
        self.annotations = annotations
        self.transform = transform

    def __getitem__(self, idx: int):
        x, y = self.images[idx], self.annotations[idx]
        x = np.load(x)['arr_0']
        y = np.load(y)['arr_0'][:,:,0]
        y[y==255] = 1

        img = Image.fromarray(y.astype('uint8')).convert('RGB')
        bbox = img.getbbox()
        height, width = y.shape[:2]
        k1 = int(random.random() * bbox[0])
        k2 = int(random.random() * bbox[1])
        k3 = int(random.random() * (width - bbox[2]) + bbox[2])
        k4 = int(random.random() * (height - bbox[3]) + bbox[3])
        y = y[k2:k4, k1:k3]
        x = x[k2:k4, k1:k3]

        if self.transform is not None:
            augmented = self.transform(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # x = x.filter(ImageFilter.EDGE_ENHANCE_MORE)

        return os.path.split(self.images[idx])[-1], \
               torch.tensor(np.array(x).astype(np.float32).transpose([2, 0, 1])), \
               torch.tensor(y.astype(np.int32)).long()

    def __len__(self):
        return len(self.images)

class TestSet2(Dataset):
    def __init__(self,
                 images,
                 annotations,
                 output_size=(420,420)):
        self.images = images
        self.annotations = annotations
        self.output_size = output_size

    def __getitem__(self, idx: int):
        oriidx=idx
        idx=idx//5
        x, y = self.images[idx], self.annotations[idx]
        x = np.load(x)['arr_0']
        y = np.load(y)['arr_0'][:,:,0]
        y[y==255] = 1

        img = Image.fromarray(y.astype('uint8')).convert('RGB')
        bbox = img.getbbox()
        height, width = y.shape[:2]
        # bbox[left, up, right, down]
        if oriidx%5==0:
            regions = [[0,0,bbox[0],bbox[1]], [bbox[0],0,bbox[2],bbox[1]], [bbox[2],0,width,bbox[1]],
                      [0,bbox[1],bbox[0],bbox[3]],[bbox[2],bbox[1],width,bbox[3]],
                      [0,bbox[3],bbox[0],height], [bbox[0],bbox[3],bbox[2],height], [bbox[2],bbox[3],width,height],]
            ks = []
            for region in regions:
                if region[2]-region[0]>self.output_size[1] and region[3]-region[1]>self.output_size[0]:
                    ks.append(self.getbox(region))
            k = ks[int(random.random()*len(ks))]
            print('ks')
            print(ks)
            print(regions)
            print('end')
        elif oriidx%5==1:
            k = [bbox[0],bbox[1],min(width, bbox[0]+height-bbox[1]),min(height, bbox[1]+width-bbox[0])]
        elif oriidx%5==2:
            k = [max(0, bbox[2]-height+bbox[1]),bbox[1],bbox[2],min(height, bbox[1]+bbox[2])]
        elif oriidx%5==3:
            k = [bbox[0], max(0, bbox[3]-width+bbox[0]), min(width, bbox[0]+bbox[3]), bbox[3]]
        elif oriidx%5==4:
            k = [max(0, bbox[2]-bbox[3]),max(0,bbox[3]-bbox[2]),bbox[2],bbox[3]]
        k = [int(t) for t in k]
        y = y[k[1]:k[3], k[0]:k[2]]
        x = x[k[1]:k[3], k[0]:k[2]]
        y = cv2.resize(y,self.output_size)
        x = cv2.resize(x,self.output_size)

        return os.path.split(self.images[idx])[-1], \
               torch.tensor(np.array(x).astype(np.float32).transpose([2, 0, 1])), \
               torch.tensor(y.astype(np.int32)).long()

    def __len__(self):
        return len(self.images) * 5

    def getbox(self, board):
        height, width = self.output_size
        x = random.random()*(board[2]-width-board[0])+board[0]
        y = random.random()*(board[3]-height-board[1])+board[1]
        return [x,y,x+min(board[2]-x,board[3]-y),y+min(board[2]-x,board[3]-y)]
