import numpy as np
from pprint import pprint
#from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import os
from statistics import NormalDist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms
#import torchmetrics
from math import log10
import copy

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

im_size=32

# tp = transforms.Compose([
#     transforms.Resize(im_size),
#     transforms.CenterCrop(im_size),
#     transforms.ToTensor()
# ])
# tt = transforms.ToPILImage()

class approx_distribution(object):
    def __init__(self,dataset=None):
        self.dataset=dataset
    

    def get_mean_std3(self, scale, row1, row2, col1, col2,w, E, multiplier):
        image_mean=[]
        for i in range(len(self.dataset)):
            #image_mean.append(torch.sum(self.dataset[i][0][:, row1:row2, col1:col2])/scale)
            a=self.dataset[i][0][:,row1:row2,col1:col2].reshape(768)
            a=torch.matmul(E.float(),a) #,0.05)
            a=torch.mul(a,w)
            a=torch.mul(a,multiplier)
            a[0:64]=0
            #a = [-8* a[i] if i % 2 != 0 else 16*a[i] for i in range(len(a))]
            # for i in range(len(a)):
            #     if i%2!=0:
            #         a[i]=-0.5*a[i]
            a=torch.tensor(a)
            image_mean.append(torch.sum(a))
        mean=np.mean(image_mean)
        std=np.std(image_mean)
        return mean,std
        
class get_bias_list(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    def create_bins(self,bin_no):
        bins=[]
        m=0
        for i in range(bin_no+1):
            m+=1/(bin_no+2)
            bins+=[NormalDist(mu=self.mean, sigma=self.std).inv_cdf(m)]
        #bins=torch.tensor(np.array(bins))
        bins = torch.sort(torch.tensor(np.array(bins)))[0]
        bin_interval = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
            #bin_sizes = bin_sizes
        return bins[:-1], bin_interval

    

class get_samples(object):
    def __init__(self,  dataset, idx):
        self.idx=idx
        self.dataset=dataset
    def get_image_label(self,scale, row1, row2, col1, col2):
        img_list=[]
        label_list=[]
        image_sum=[]
        for i in range(len(self.idx)):
            #print(self.dataset[self.idx[i]])
            #print(type(self.dataset[0][0]))
            #print(self.dataset[0][0].shape)
            data_point = self.dataset[self.idx[i]][0][:,row1:row2,col1:col2].to(device)
            data_label = torch.Tensor([self.dataset[self.idx[i]][1]]).long().to(device)
    #print(gt_label)
            data_point = data_point.view(1, *data_point.size())
            data_label = data_label.view(1, )
    #print(gt_label)
            #data_onehot_label = label_to_onehot(gt_label2, num_classes=num_classes)
    
            img_list.append(data_point.detach().clone())
            label_list.append(data_label.detach().clone())
            image_sum.append(torch.sum(data_point).item())

        data_dist = []

        for i in range(len(label_list)):
            data_dist.append(image_sum[i]/(scale))

        data  = torch.cat(img_list)
#print(gt_data2.shape)
        label = torch.cat(label_list)
        return data, label, data_dist