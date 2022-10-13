import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, models
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import cv2
import torchvision.transforms as transforms
import numpy as np
import timm

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CFG={
    'EPOCHS':15,
    'LR':1e-4
}

file_name=['Arborio','Basmati','Ipsala','Jasmine','Karacadag']
img_list={"target":[],"idx":[]}
for name in file_name:
  img_names=os.listdir(f"./Rice_Image_Dataset/{name}")
  for img_name in tqdm(img_names):
    img=img_name.split(").")
    img=img[0].split("(")
    img_list["target"].append(name)
    img_list["idx"].append(int(img[1]))
img_list["idx"]=np.nan_to_num(img_list["idx"])

def aug_random_imshow(idx,transform):
  global file_name
  plt.figure(figsize=(10,10))
  c+=0
  for name in file_name:
    c+=1
    plt.subplot(1,8,c)
    image=cv2.imread("./Rice_Image_Dataset/{0}/{0} ({1}).jpg".format(name, idx))
    augmentations = transform(image=image)
    plt.show(image)
    plt.title(f'name original')
    
    c+=1
    plt.subplot(1,8,c)
    plt.show(augmentations)
    plt.title(f'{name} transform')

  plt.show()

height = 224
width = 224
transform = A.Compose([
    A.Normalize(),
    A.Resize(height=224, width=224),
    A.RandomResizedCrop(height=224, width=224, scale=(0.3, 1.0)),
])

aug_random_imshow(10,transform)
augmentations = transform(image=image, mask=mask)

class CustomDataset(Dataset):
  def __init__(self,img_list,img_target,label,transforms):
    self.img_list=img_list
    self.transforms=transforms
    self.img_target=img_target
    self.label=label

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self,idx):
    index=self.img_list[idx]
    label=self.img_target[index]
    image=cv2.imread("./Rice_Image_Dataset/{0}/{0} ({1}).jpg".format(label, index))
    image=np.asarray(image,dtype=np.uint8)
    image=Image.fromarray(image.astype(np.uint8))
    label=self.label[index]
    if self.transforms:
      image=self.transforms(image)
    label=torch.tensor(label)
    return image.clone().detach(),label

train,test=train_test_split(img_list['idx'], test_size=0.2, shuffle=True, stratify=img_list['target'],random_state=34)
train,valid=train_test_split(train, test_size=0.2, shuffle=True, random_state=34)

img_list['target']
#targets[15111]

transforms_train=transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

label=img_list['target']
le = preprocessing.LabelEncoder()
targets = le.fit_transform(label)

train_dataset=CustomDataset(train,img_list['target'],targets,transforms=transforms_train)
val_dataset=CustomDataset(valid,img_list['target'],targets,transforms=transforms_train)

train_dataloader=torch.utils.data.DataLoader(
    train_dataset,
    pin_memory=True,
    batch_size=32,
    num_workers=4,
    shuffle=True
)
valid_dataloader=torch.utils.data.DataLoader(
    val_dataset,
    pin_memory=True,
    batch_size=32,
    num_workers=4,
    shuffle=True
)

class BaseModel(nn.Module):
  def __init__(self,num_classes=5):
    super(BaseModel,self).__init__()
    self.model=timm.create_model('efficientnet_b0',pretrained=True,num_classes=5) #in_chans

  def forward(self,x):
    x=self.model(x)
    return x

def validation(model, optimizer,device,criterion, scheduler=None):
  best_score=0
  model.eval()
  model_preds=[]
  true_labels=[]
  valid_loss=[]
  with torch.no_grad():
    for data,label in tqdm(iter(valid_dataloader)):
      data,label=data.to(device),label.to(device)
      
      output=model(data)
      loss=criterion(output,label)
      valid_loss.append(loss.item())
      
      model_preds+=output.argmax(1).detach().cpu().numpy().tolist()
      true_lables+=label.detach().cpu().numpy().tolist()

  return np.mean(valid_loss), accuracy_score(true_labels,model_preds)

def train(model, optimizer,device,criterion, scheduler=None):
  for epoch in range(1,CFG['EPOCHS']+1):
    model.train()
    train_loss=[]
    true_label=[]
    model_preds=[]
    for data,label in tqdm(iter(train_dataloader)):
      data,label=data.to(device),label.to(device)
      optimizer.zero_grad()
      
      output=model(data)
      loss=criterion(output,label)
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
      if scheduler is not None:
        scheduler.step()
      model_preds+=output.argmax(1).detach().cpu().numpy().tolist()
      true_label+=label.detach().cpu().numpy().tolist()     
    val_loss,val_acc=validation(model, optimizer,device,criterion, scheduler)
    print(f'Epoch:{epoch} Train Loss : {np.mean(train_loss)} Valid Loss : {val_loss} Valid Acc : {val_acc}')

file_name=['Arborio','Basmati','Ipsala','Jasmine','Karacadag']

device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model=BaseModel().to(device)
optimizer=torch.optim.SGD(model.parameters(), lr=CFG['LR'])
criterion=nn.CrossEntropyLoss().to(device)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.001)

train(model,optimizer,device,criterion,scheduler)

