#!/usr/bin/env python
# coding: utf-8

#model_for_submission.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import distance_transform_edt, gaussian_filter
import bz2
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
import random
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import models_utils as mu
from sklearn.model_selection import train_test_split
import cv2
import time
from livelossplot import PlotLosses
import torchvision.transforms.functional as TF
import pandas as pd
from collections import Counter
import ASPP_utils as amu

df = pd.read_csv('final_dataset_filtered_cleaned.csv') 

df['name'] = df['name'].apply(lambda x: x if x in ['Musacea','Guaba','Cacao','Mango'] else 'Otra variedad')

# Define the transform pipeline
transform = amu.RandomTransforms(
    horizontal_flip_prob=0.5,
    vertical_flip_prob=0.5,
    rotation_degrees=[0, 90, 180, 270]
)

test_transform = transforms.Compose([
    transforms.ToTensor()
])

#split into train, test, and val 
unique_itcs = df['unique_id'].unique()
unique_df = df.drop_duplicates(subset='unique_id')[['unique_id', 'name']]
train_itcs, test_itcs = train_test_split(
    unique_itcs, test_size=0.4, random_state=42, stratify=unique_df['name']
)
train_itcs, val_itcs = train_test_split(
    train_itcs, test_size=0.2, random_state=42, stratify=unique_df[unique_df['unique_id'].isin(train_itcs)]['name']
)

test_df = df[df['unique_id'].isin(test_itcs)]
#save test data for later evaluation 
test_df.to_csv('test_data_ASPP.csv', index=False)
train_df = df[df['unique_id'].isin(train_itcs)]
val_df = df[df['unique_id'].isin(val_itcs)]

print(f'Training set size: {len(train_df_small)}')
print(f'Validation set size: {len(val_df_small)}')
print(f'Testing set size: {len(test_df_small)}')

root_dir = 'data/tiles/processed'

train_dataset = amu.TreeCrownDataset(dataframe=train_df, root_dir=root_dir, transform=transform)
val_dataset = amu.TreeCrownDataset(dataframe=val_df, root_dir=root_dir, transform=transform)
test_dataset = amu.TestDataset(dataframe=test_df, root_dir=root_dir, transform=None)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("All datasets loaded successfully")

# Analyze class distribution in the test set
class_distribution = amu.analyze_class_distribution(val_loader)
print("Class distribution in the val set:", class_distribution)

device = amu.set_device("cuda")

model = amu.SEDDModel(num_classes=6).to(device)
semantic_loss_fn = amu.PartiallyWeightedCategoricalFocalLoss(alpha=0.25).to(device)
distance_loss_fn = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 10

for epoch in range(num_epochs):
    logs = {}
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    train_loss = amu.fit(model, train_loader, optimizer, semantic_loss_fn, distance_loss_fn, device)
    val_loss = amu.validate(model, val_loader, semantic_loss_fn, distance_loss_fn, device)
    
    scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), f'ASPP_model{epoch+1}.pth')

model = amu.SEDDModel(num_classes=6).to(device)
model.load_state_dict(torch.load('ASPP_model10.pth'))

model.eval()

patch_size = (224, 224)  # Patch size
stride = (112, 112)  # 50% overlap

accuracy, precision, recall, f1, mse = amu.evaluate_sliding_window(model, test_loader, patch_size, stride, device)

print("Results with 50% overlap:")
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print(f'Test Distance Map MSE: {mse:.4f}')

#set to 30% overlap
stride = (67, 67)  # 30% overlap

accuracy, precision, recall, f1, mse = amu.evaluate_sliding_window(model, test_loader, patch_size, stride, device)

print("Results with 30% overlap:")
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print(f'Test Distance Map MSE: {mse:.4f}')

#set to 10% overlap
stride = (22, 22)  # 10% overlap

accuracy, precision, recall, f1, mse = amu.evaluate_sliding_window(model, test_loader, patch_size, stride, device)

print("Results with 10% overlap:")
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print(f'Test Distance Map MSE: {mse:.4f}')
