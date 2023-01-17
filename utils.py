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
from typing import Union, Tuple, List


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


def save_mean_and_std(mean, std, filename):
    """
    Thies functions takes the mean and standard deviation of the dataset
    and save it to a JSON file.
    
    ...
    Args:
        mean: list or float 
            A single or a list of float values represent the mean of the dataset 
        std: list or float 
            A single or a list of float values represent the standard deviation
            of the dataset 
        filename: str
            A string value that represents the file name where the data will be
            saved into
    ...
    
    """
    
    file_path = os.path.join(os.getcwd(), filename)
    stats = {"mean": mean,
             "std": std}

    with open(file_path, 'w') as file:
        file.write(json.dumps(stats))


def calculate_mean_and_std(data_loader):
    """
    A function that calculates the mean and standard deviation of the dataset
    
    ...
    Args:
        data_loader: tourch.utils.data.DataLoader
            A data loader that provides the data for calculation
    ...
    """
    
    mean, std = 0., 0.
    num_samples = 0
    
    for imgs, _ in data_loader:
        imgs_in_a_batch = imgs.size(0)
        imgs = imgs.view(imgs_in_a_batch, 3, -1)
        mean += imgs.mean(-1).sum(0)
        std += imgs.std(-1).sum(0)
        num_samples += imgs_in_a_batch
    
    mean = (mean / num_samples).tolist()
    std = (std / num_samples).tolist()
    return mean, std


def get_mean_and_std(data_loader, filename, redo=False):
    """
    A function that retreives the mean and standard deviation of the dataset
    from the saved JSON file if exict. if not, it call the calculate_mean_and_std
    function to calculate the mean and std
    
    ...
    Args:
        data_loader: tourch.utils.data.DataLoader
            A data loader that provides the data for calculation
        filename: str
            The JSON file that contains the calculated mean and std
        redo: boolean
            If True -> calculate the mean and std even if they were saved to JSON file
            if False -> retreive the saved values if they exict. otherwise, calculate them
    ...
    """
    stats_dict = None
    if os.path.exists(filename) == False or redo == True:
        mean, std = calculate_mean_and_std(data_loader)
        save_mean_and_std(mean, std, filename)
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
    

def split_samples(num_samples : int,
                  ratio : Union[float,
                                List[float],
                                Tuple[float]]):
    """
    A function that calculates the number of samples used for training
    and validation with respect to a certain ration
    
    ...
    Args:
        num_samples: int
            The total number of samples in the dataset
        ratio: float or (list, tuple) of size 2
            The proportion used to specify the numer of samples used for training
            For training: num_samples * ratio
            For validation: num_samples - training_samples
    
    ...
    
    """
    if isinstance(ratio, (list, tuple)):
        train_samples = int(num_samples * ratio[0])
    else:
        train_samples = int(num_samples * ratio)
        
    validation_samples = num_samples - train_samples
    return (train_samples, validation_samples)


def predictions(model, data_loader, device):
    """
    A function that collect the predicted labels estimated by the BrainTumorClassifier
    for each corresponding true label
    
    ...
    Args:
        model: BrainTumorClassifier
            A trained instance of the BrainTumorClassifier
        data_loader: torch.utils.data.DataLoader
            A data loader that provide the imgs and true labels for the model
        device: str
            A CUDA GPU or CPU
    
    ...
    
    """
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
    """
    A function that returns the confusion matrix returned by 
    confusion_matrix(y_true, y_pred) from scikit-learn
    
    ...
    
    Args:
        y_true: list
            A list of integers contains the true labels of the corresponding images
        Y_pred : list
            A list of integers contains the labels predicted by the BrainTumorClassifier

    ...
    
    """
    
    return confusion_matrix(y_true, y_pred)


def plot_cf_matrix(cf_matrix, classes):
    """
    A function used to plot the confusion matrix as a heatmap table
    
    ...
    
    Args:
        cf_matrix: ndarray
            The confusion matrix generated by 
            confusion_matrix(y_true, y_pred) from scikit-learn
        classes : list
            A list of strings containing the class labels
    
    ...
    
    """
    
    cf_matrix_dataFrame = pd.DataFrame(cf_matrix, index = classes, columns = classes)
    fig, ax = plt.subplots()    
    plt.title("Confusion Matrix")
    sns.heatmap(cf_matrix_dataFrame, annot=True, fmt='.0f')
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.show()
    
 
def get_classification_report(y_true, y_pred, labels=None, target_names=None):
    """
    A function that returns the classification report generted from scikit-learn
    using the true and predicted labels
    
    ...
    
    Args:
        y_true: list
            A list of integers contains the true labels of the corresponding images
        Y_pred : list
            A list of integers contains the labels predicted by the BrainTumorClassifier
        labels: list
            An optional list of label indices to include in the report
        target_names:
            An optional list of string containing the names of classes matching the labels

    ...
    
    """
    return classification_report(y_true=y_true,
                                 y_pred=y_pred,
                                 labels=labels,
                                 target_names=target_names)  
     
    
def plot_per_class_accuracy(cf_matrix, classes):
    """
    A function used to plot the classification accuracy for each class
    
    ...
    
    Args:
        cf_matrix: ndarray
            The confusion matrix generated by 
            confusion_matrix(y_true, y_pred) from scikit-learn
        classes : list
            A list of strings containing the class labels
    
    ...
    
    """
    accs = cf_matrix.diagonal() / cf_matrix.sum(1)    
    fig, ax = plt.subplots()
    bars = ax.bar(classes, accs)
    ax.bar_label(bars)
    plt.title('Class Accuracy')
    plt.ylabel("Accuracy %")
    plt.xlabel("Class Label")
    plt.show()
    

