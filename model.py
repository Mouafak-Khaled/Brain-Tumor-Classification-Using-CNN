import os
import torch
import numpy as np
import torch.nn as nn

class BrainTumorClasssifer(nn.Module):
    
    def __init__(self, in_channels, num_classes, bias=False):
        super(BrainTumorClasssifer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bias = bias
        
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.dropout2d = nn.Dropout2d(p=0.5)
        
        self.conv = nn.Conv2d(self.in_channels, 128, kernel_size=3, stride=(2, 2), bias=self.bias)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        self.bn = nn.BatchNorm2d(128)

        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), bias=self.bias)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, bias=self.bias)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, bias=self.bias)
        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(1024, 512, bias=self.bias)
        self.ln1 = torch.nn.LayerNorm(512)
        
        self.fc2 = nn.Linear(512, 256, bias=self.bias)
        self.ln2 = torch.nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, self.num_classes)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
    
    
    def forward(self, x):
        
        x = self.relu(self.bn(self.conv(x)))
        x = self.relu(self.bn1(self.mp1(self.conv1(x))))   
        x = self.relu(self.bn2(self.mp2(self.conv2(x))))
        x = self.relu(self.bn3(self.mp3(self.conv3(x))))

        x = self.flatten(x)
        
        x = self.relu(self.dropout(self.ln1(self.fc1(x))))
        x = self.relu(self.dropout(self.ln2(self.fc2(x))))
        
        x = self.fc3(x)
        return x


    
        