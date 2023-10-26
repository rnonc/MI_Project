# Write by Rodolphe Nonclercq
# September 2023
# ILLS-LIVIA
# contact : rnonclercq@gmail.com


import torch
import matplotlib.pyplot as plt
import torchvision

class data_augm:
    def __init__(self,resolution):
        self.H_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.Jitter = torchvision.transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1))
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)
    def transform(self,x):
        x = self.resize(x)
        x = self.H_flip(x)
        x = self.Jitter(x)
        x = x/255
        #x = (x-torch.mean(x))/torch.std(x)
        return x

class data_adapt:
    def __init__(self,resolution):
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)
    def transform(self,x):
        x = self.resize(x)
        x = x/255
        #x = (x-torch.mean(x))/torch.std(x)
        return x
