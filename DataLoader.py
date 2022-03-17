import os
import pandas as pd
import torch
from torch.utils import data
from PIL import Image

class ImageFolder(data.Dataset):
  def __init__(self, image_folder, gt_folder, name_list, transform=None):
    self.image_folder = image_folder
    self.gt_folder = gt_folder
    self.name_list = name_list
    self.transform = transform
    
  def __getitem__(self, index):
    name = self.name_list[index]
    
    img_path = os.path.join(self.image_folder, name)
    gt_path = os.path.join(self.gt_folder, name)
    
    image = Image.open(img_path)
    ground_truth = Image.open(gt_path)
    
    if self.transform:
      image, ground_truth = self.transform(image, ground_truth)
      
    if ground_truth.shape[0] == 3:
      ground_truth = ground_truth[0,:,:]
      ground_truth = torch.unsqueeze(ground_truth, 0)
      
    return image, ground_truth
    
  def __len__(self):
    return len(self.name_list)


def get_loader(image_folder, gt_folder, name_list, batch_size, transform, shuffle=True, num_workers=2, drop_last=False):
  dataset = ImageFolder(image_folder=image_folder,
                        gt_folder=gt_folder,
                        name_list=name_list,
                        transform=transform)
  
  data_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=drop_last)
  return data_loader