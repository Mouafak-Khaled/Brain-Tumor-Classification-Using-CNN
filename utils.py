import os
import json
import torch
import random
import numpy as np
from torchvision import transforms as T
import torchvision.transforms.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


IMG_SIZE = 256


def get_device():
    """
    Checks if a CUDA device is available
        - if yes, return a cuda device
        - if no, return a cpu device
    """
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


def save_model(model, param_name, path):
    """
    Args:
        - model : a machine learning model that we want to save
        - param_name: the name to be used for the saved model
        - path: the path to the destination where we want to save the model
    """
    path = os.path.join(path, param_name)
    with open(path, 'wb') as file:
        torch.save({'model_state_dict': model.state_dict()}, file)


def load_model(param_name, path):
    """
    Args:
        - param_name: the name of the model parameters file
        - path: the path where the model parameters exists
    return the model state dictionary
    """
    path = os.path.join(path, param_name)
    with open(path, 'rb') as file:
        model_state_dict = torch.load(file)['model_state_dict']
    return model_state_dict


def _write_to_a_json_file(mean, std, filename):

    file_path = os.path.join(os.getcwd(), filename)

    stats = {"mean": mean,
             "std": std}

    with open(file_path, 'w') as file:
        file.write(json.dumps(stats))


def calculate_mean_and_std(dataset, filename):
    channels_sum, channels_squared_sum= 0, 0
    N = len(dataset)
    for img, _ in dataset:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(img, dim=[1,2])
        channels_squared_sum += torch.mean(img ** 2, dim=[1, 2])
    
    mean = channels_sum / N
    std = (channels_squared_sum / N - mean ** 2) ** 0.5 
    _write_to_a_json_file(mean.tolist(), std.tolist(), filename)
    return mean , std


def calculate_mean_and_std_2(data_loader, N, img_size, filename):
    channels_sum, channels_squared_sum= 0, 0
    N = N * img_size * img_size 
    for imgs, _ in data_loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += imgs.sum(dim=[0, 2 , 3])
        channels_squared_sum += (imgs ** 2).sum(dim=[0, 2, 3])
    
    mean = channels_sum / N
    std = (channels_squared_sum / N - mean ** 2) ** 0.5 
    _write_to_a_json_file(mean.tolist(), std.tolist(), filename)
    return mean , std


        
def get_mean_and_std(dataset, N, img_size, filename, redo=False):
    stats_dict = None
    if os.path.exists(filename) == False or redo == True:
        mean, std = calculate_mean_and_std_2(dataset,N, img_size, filename)
    else:
        with open(filename, 'r') as json_file:
            stats_dict = json.loads(json_file.read())
            
        mean, std = stats_dict['mean'], stats_dict['std']
    mean= torch.round(torch.FloatTensor(mean), decimals=4)
    std = torch.round(torch.FloatTensor(std), decimals=4)
    return mean, std
    
    
def get_augment_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.7),
        T.RandomAutocontrast(p=0.6),
        T.RandomRotation(degrees=45)])
    
    
def get_transforms(mean=None, std=None):
    
    trans_dict = {
        # 'resize':T.Resize([224, 224]),
        'ToTensor': T.ToTensor(),
        'Normalize': None}
    
    if mean != None and std != None:
        trans_dict['Normalize'] = T.Normalize(mean=mean, 
                                              std=std)
    
    transforms = T.Compose([trans_dict[key] for key in trans_dict if trans_dict[key] != None])
    
    return transforms


def split_samples(num_samples : int, ratio : float):
    
    train_samples = int(num_samples * ratio)
    validation_samples = num_samples - train_samples
    return (train_samples, validation_samples)


def custom_lr_factor(epoch):
    if epoch < 8:
        return 1
    elif epoch >=8 and epoch < 15:
        return 0.5
    else:
        return 0.2 


def predictions(model, data_loader, device):
    
    y_pred, y_true = [], []

    for imgs, labels in data_loader:
        
        with torch.set_grad_enabled(False):
            
            imgs = imgs.to(device, dtype=torch.float)
            labels = labels.type(torch.LongTensor)

            labels = labels.to(device)
            
            output = model(imgs)

            _, output = torch.max(output, 1)
            y_pred.extend(output.data.cpu().numpy()) 
            y_true.extend(labels.data.cpu().numpy())
    
    return y_true, y_pred
     
    
def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def plot_cf_matrix(cf_matrix, classes):
    
    
    cf_matrix_dataFrame = pd.DataFrame(cf_matrix, index = classes, columns = classes)
    fig, ax = plt.subplots()    
    plt.title("Confusion Matrix")
    sns.heatmap(cf_matrix_dataFrame, annot=True, fmt='.0f')
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 

    plt.show()
    
 
def get_classification_report(y_true, y_pred, labels=None, target_names=None):
    
    return classification_report(y_true=y_true,
                                 y_pred=y_pred,
                                 labels=labels,
                                 target_names=target_names)  
     
    
def plot_per_class_accuracy(cf_matrix, classes):
    
    accs = cf_matrix.diagonal() / cf_matrix.sum(1)    
    fig, ax = plt.subplots()
    bars = ax.bar(classes, accs)
    ax.bar_label(bars)
    plt.title('Class Accuracy')
    plt.ylabel("Accuracy %")
    plt.xlabel("Class Label")
    plt.show()
    
###########################################################################


#   def augment(self):
#         if self.augment_transform==None:
#             pass
#         else:
#             for mode in MODE:
#                 if mode == 'train':
#                     data_path = os.path.join(self.data_path, mode)
#                     if os.path.exists(data_path):
#                         path_list = os.listdir(data_path)
#                     for label, p in enumerate(path_list):
#                         count = 0
#                         path = os.path.join(data_path, p)
#                         image_list = os.listdir(path)
#                         N = len(image_list)
#                         max_number_of_samples = 2500
                        

#                         while N != max_number_of_samples:
#                             img_path = random.choice(image_list)
#                             img = Image.open(os.path.join(path, img_path)).convert('RGB')
#                             img = self.augment_transform(img)
#                             save_image(T.ToTensor()(img), os.path.join(path, self.mode + '_au' + str(label) + str(count + 1) + '.jpg'))
#                             count += 1
#                             N +=1   
