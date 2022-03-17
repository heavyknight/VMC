import os
import numpy as np
from datetime import datetime

class File_manager(object):
  def __init__(self, root, model, dataset):
    self.root = root
    self.filename = datetime.now().strftime(f'{model}_{dataset}_%d%b%Y_%H:%M')
    self.filepath = os.path.join(self.root, self.filename)
    
  def build_file(self):
    if os.path.exists(self.filepath):
      print(f'{self.filename} already exists in {self.root}')
    else:
      os.makedirs(self.filepath)
      print(f'{self.filename} has been created in {self.root}')
      
  def get_filename(self):
    return self.filename
    
  def get_filepath(self):
    return self.filepath
    
  def get_root(self):
    return self.root


class Metric(object):
  def __init__(self, name):
    self.name = name
    self.epochs = 0
    self.iter_stats = self.new_stats()
    self.epoch_stats = self.new_stats()
    self.tmp_stats = self.new_stats()
    
  def epoch(self):

    self.epochs += 1

    for key in self.iter_stats.keys():
      self.iter_stats[key] += self.tmp_stats[key]

    out = self.new_out()

    for key in self.epoch_stats.keys():
      value = np.mean(self.tmp_stats[key])
      self.epoch_stats[key].append(value)
      out[key] = np.round(value, 3)

    self.tmp_stats = self.new_stats()

    return out

  def new_out(self):
    tmp_out = dict(epoch=self.epochs,
                   loss=0.0,
                   accuracy=0.0,
                   sensitivity=0.0,
                   specificity=0.0,
                   precision=0.0,
                   dice=0.0,
                   IoU=0.0)

    return tmp_out

  def new_stats(self):
    tmp_stats = dict(loss=list(),
                     accuracy=list(),
                     sensitivity=list(),
                     specificity=list(),
                     precision=list(),
                     dice=list(),
                     IoU=list())
    return tmp_stats
      
  def calculate_current_stats(self, prediction, ground_truth, loss=0.0):
    if prediction.dim() > 2: 
      prediction = prediction.view(prediction.size(0), -1)
    if ground_truth.dim() > 2: 
      ground_truth = ground_truth.view(ground_truth.size(0), -1)

    threshold = 0.5
    a_ = 1e-6
    
    prediction = prediction > threshold
    ground_truth = ground_truth > threshold
    
    TP = (((prediction == 1).int() + (ground_truth == 1).int()) == 2).float().sum(1)
    FN = (((prediction == 0).int() + (ground_truth == 1).int()) == 2).float().sum(1)
    TN = (((prediction == 0).int() + (ground_truth == 0).int()) == 2).float().sum(1)
    FP = (((prediction == 1).int() + (ground_truth == 0).int()) == 2).float().sum(1)
    Inter = ((prediction.int() + ground_truth.int()) == 2).float().sum(1)
    Union = ((prediction.int() + ground_truth.int()) >= 1).float().sum(1)
    
    accuracy = (TP+TN)/ground_truth.size(1)
    sensitivity = TP/(TP+FN+a_)
    specificity = TN/(TN+FP+a_)
    precision = TP/(TP+FP+a_)
    dice = (2*sensitivity*precision)/(sensitivity+precision+a_)
    IoU = Inter/(Union+a_)
    
    accuracy = accuracy.mean().item()
    sensitivity = sensitivity.mean().item()
    specificity = specificity.mean().item()
    precision = precision.mean().item()
    dice = dice.mean().item()
    IoU = IoU.mean().item()
    
    self.tmp_stats["loss"].append(loss)
    self.tmp_stats["accuracy"].append(accuracy)
    self.tmp_stats["sensitivity"].append(sensitivity)
    self.tmp_stats["specificity"].append(specificity)
    self.tmp_stats["precision"].append(precision)
    self.tmp_stats["dice"].append(dice)
    self.tmp_stats["IoU"].append(IoU)

    
  def save(self, filepath="./", filename=None):
    if filename == None:
      filename = f"{self.name}.npy"
    parameters = dict(iter_stats=self.iter_stats,
                      epoch_stats=self.epoch_stats,
                      name=self.name,
                      epochs=self.epochs)
    root = os.path.join(filepath, filename)
    np.save(root, parameters)
    
  def load(self, path):
    parameters = np.load(path, allow_pickle=True).item()
    if isinstance(parameters, dict):
      self.iter_stats = parameters["iter_stats"]
      self.epoch_stats = parameters["epoch_stats"]
      self.name = parameters["name"]
      self.epochs = parameters["epochs"]


class TestMetric(object):
  def __init__(self, name):
    self.histogram = None
    self.statistics = dict()
    self.name = name

  def make_histogram(self):
    self.histogram = dict(loss=list(),
                          accuracy=list(),
                          sensitivity=list(),
                          specificity=list(),
                          precision=list(),
                          dice=list(),
                          IoU=list())
    
    for key in self.statistics.keys():
      for param in self.statistics[key].keys():
        self.histogram[param].append(self.statistics[key][param])
      
  def get_test_summary(self):
    summary = dict(loss=0.0,
                   accuracy=0.0,
                   sensitivity=0.0,
                   specificity=0.0,
                   precision=0.0,
                   dice=0.0,
                   IoU=0.0)

    if self.histogram==None:
      self.make_histogram()

    for key in self.histogram.keys():
      mean_value = np.mean(self.histogram[key])
      summary[key] = np.round(mean_value, 3)
    return summary
      
  def calculate_stats(self, img_name, prediction, ground_truth, loss=0.0):
    if prediction.dim() > 2: prediction = prediction.view(prediction.size(0), -1)
    if ground_truth.dim() > 2: ground_truth = ground_truth.view(ground_truth.size(0), -1)
    threshold = 0.5
    a_ = 1e-6
    
    prediction = prediction > threshold
    ground_truth = ground_truth > threshold
    
    TP = (((prediction == 1).int() + (ground_truth == 1).int()) == 2).float().sum(1)
    FN = (((prediction == 0).int() + (ground_truth == 1).int()) == 2).float().sum(1)
    TN = (((prediction == 0).int() + (ground_truth == 0).int()) == 2).float().sum(1)
    FP = (((prediction == 1).int() + (ground_truth == 0).int()) == 2).float().sum(1)
    Inter = ((prediction.int() + ground_truth.int()) == 2).float().sum(1)
    Union = ((prediction.int() + ground_truth.int()) >= 1).float().sum(1)
    
    accuracy = 100.0*(TP+TN)/ground_truth.size(1)
    sensitivity = 100.0*(TP)/(TP+FN+a_)
    specificity = 100.0*(TN)/(TN+FP+a_)
    precision = 100.0*(TP)/(TP+FP+a_)
    dice = (2*sensitivity*precision)/(sensitivity+precision+a_)
    IoU = 100.0*Inter/(Union+a_)
    
    accuracy = accuracy.mean().item()
    sensitivity = sensitivity.mean().item()
    specificity = specificity.mean().item()
    precision = precision.mean().item()
    dice = dice.mean().item()
    IoU = IoU.mean().item()

    tmp = dict(loss=0.0,
               accuracy=0.0,
               sensitivity=0.0,
               specificity=0.0,
               precision=0.0,
               dice=0.0,
               IoU=0.0)
    
    tmp["loss"] = loss
    tmp["accuracy"] = accuracy
    tmp["sensitivity"] = sensitivity
    tmp["specificity"] = specificity
    tmp["precision"] = precision 
    tmp["dice"] = dice
    tmp["IoU"] = IoU

    self.statistics.update({img_name : tmp})
    
  def save(self, filepath="./", name=None):
    if name == None:
      name = f"{self.name}.npy"
    filepath = os.path.join(filepath,name)
    parameters = dict(histogram=self.histogram,
                      statistics=self.statistics,
                      name=self.name)
    np.save(filepath, parameters)
    
  def load(self, path):
    parameters = np.load(path, allow_pickle=True).item()
    if isinstance(parameters, dict):
      self.histogram = parameters["histogram"]
      self.statistics = parameters["statistics"]
      self.name = parameters["name"]


def save_config(path, model_name, criterion, optimizer, batch_size, lr, beta1, beta2, weight_decay, epochs, train_transform):
    info = f'Model = {model_name}\n'
    info += f'Loss = {criterion}\n'
    info += f'Optimizer = {optimizer}\n'
    info += f'batch = {batch_size}\n'
    info += f'lr = {lr}\n'
    info += f'beta1 = {beta1}\n'
    info += f'beta2 = {beta2}\n'
    info += f'weight decay = {weight_decay}\n'
    info += f'epochs = {epochs}\n'

    t_tr = 'train transform: '+format(train_transform)

    info += t_tr
    resultpath = os.path.join(path, 'config.txt')
    with open(resultpath, 'w') as f:
        f.write(info)
