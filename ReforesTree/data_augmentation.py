#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import distance_transform_edt
from livelossplot import PlotLosses

# Loading in the data
df = pd.read_csv('final_dataset_filtered.csv')
print('Data has been loaded')

# Augmenting the Data
def random_rotation(image):
    rotations = [0, 90, 180, 270]
    angle = random.choice(rotations)
    return transforms.functional.rotate(image, angle)

# Creating the Transform Pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Lambda(lambda img: random_rotation(img)),
    transforms.ToTensor()
])

# Generating Distance Maps
def generate_distance_map(image_shape, bbox):
    distance_map = np.zeros(image_shape, dtype=np.float32)
    xmin, ymin, xmax, ymax = bbox.int().tolist()
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 1
    distance_map = distance_transform_edt(mask == 0)
    distance_map = distance_map / distance_map.max()  # Normalize distances to [0, 1]
    return distance_map

# Custom Dataset
class TreeCrownDataset(Dataset):
    def __init__(self, dataframe, root_dir, split, transform=None, val_size=0.15, test_size=0.15, random_state=42):
        self.root_dir = root_dir
        self.transform = transform

        # Split the dataframe into train, validation, and test sets
        train_val_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=random_state)
        train_df, val_df = train_test_split(train_val_df, test_size=val_size / (1 - test_size), random_state=random_state)

        if split == 'train':
            self.dataframe = train_df
        elif split == 'val':
            self.dataframe = val_df
        elif split == 'test':
            self.dataframe = test_df
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load image
        img_path = os.path.join(self.root_dir, row['img_path'])
        image = Image.open(img_path).convert('RGB')

        # Get metadata
        species = row['name']
        species_label = torch.tensor(0 if species == 'Musacea' else 1 if species == 'Guaba' else 2 if species == 'Cacao' else 3 if species == 'Mango' else 4, dtype=torch.long)
        carbon = torch.tensor(row['carbon'], dtype=torch.float32)
        bbox = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        # Generate distance map on the fly
        distance_map = generate_distance_map(image.size()[1:], bbox)
        distance_map = torch.from_numpy(distance_map).unsqueeze(0)  # Add channel dimension

        return image, species_label, distance_map, bbox, carbon

root_dir = 'data/tiles/processed'

# Loading and Visualizing Data
df_subset = df.sample(frac=0.1, random_state=42)

# Using a smaller dataset for memory constraints
train_dataset = TreeCrownDataset(dataframe=df_subset, root_dir=root_dir, split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_dataset = TreeCrownDataset(dataframe=df_subset, root_dir=root_dir, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)
val_dataset = TreeCrownDataset(dataframe=df_subset, root_dir=root_dir, split='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)

print('Dataloaders have been rendered')

# Setting Device
def set_device(device, idx=0):
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print("Cuda installed! Running on GPU {} {}!".format(idx, torch.cuda.get_device_name(idx)))
            device="cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print("Cuda installed but only {} GPU(s) available! Running on GPU 0 {}!".format(torch.cuda.device_count(), torch.cuda.get_device_name()))
            device="cuda:0"
        else:
            device="cpu"
            print("No GPU available! Running on CPU")
    return device

device = set_device()

# Establishing the Model Architecture
class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove last layers

    def forward(self, x):
        return self.resnet(x)

class SemanticSegmentationDecoder(nn.Module):
    def __init__(self, num_classes):
        super(SemanticSegmentationDecoder, self).__init__()
        self.deeplab_head = DeepLabHead(512, num_classes)  

    def forward(self, x):
        return self.deeplab_head(x)

class DistanceMapDecoder(nn.Module):
    def __init__(self):
        super(DistanceMapDecoder, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)  # Single channel output for distance map

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.Sigmoid()(self.conv2(x))
        return x

class PartiallyWeightedCategoricalFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, beta=0.75):
        super(PartiallyWeightedCategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        # Compute cross entropy
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)

        # Compute focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Compute the weights
        weights = torch.where(target == 0, self.alpha, self.beta)

        # Apply the weights
        loss = weights * focal_loss

        return loss.mean()

class SEDDModel(nn.Module):
    def __init__(self, num_classes):
        super(SEDDModel, self).__init__()
        self.encoder = ResNet18Encoder()
        self.semantic_decoder = SemanticSegmentationDecoder(num_classes)
        self.distance_decoder = DistanceMapDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        semantic_output = self.semantic_decoder(encoded)
        distance_output = self.distance_decoder(encoded)
        return semantic_output, distance_output

def final_loss(semantic_loss, distance_loss):
    return semantic_loss + distance_loss

def fit(model, dataloader, optimizer, semantic_loss_fn, distance_loss_fn):
    model.train()
    running_loss = 0.0

    for images, semantic_targets, distance_targets, _, _ in dataloader:
        images = images.to(device)
        semantic_targets = semantic_targets.to(device)
        distance_targets = distance_targets.to(device)

        optimizer.zero_grad()
        semantic_outputs, distance_outputs = model(images)

        semantic_loss = semantic_loss_fn(semantic_outputs, semantic_targets)
        distance_loss = distance_loss_fn(distance_outputs, distance_targets)
        loss = final_loss(semantic_loss, distance_loss)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader.dataset)
    return train_loss

def validate(model, dataloader, semantic_loss_fn, distance_loss_fn):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, semantic_targets, distance_targets, _, _ in dataloader:
            images = images.to(device)
            semantic_targets = semantic_targets.to(device)
            distance_targets = distance_targets.to(device)

            semantic_outputs, distance_outputs = model(images)
            semantic_loss = semantic_loss_fn(semantic_outputs, semantic_targets)
            distance_loss = distance_loss_fn(distance_outputs, distance_targets)
            loss = final_loss(semantic_loss, distance_loss)

            running_loss += loss.item()

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss

# Training the Model
num_epochs = 2
model = SEDDModel(num_classes=5).to(device)
semantic_loss_fn = PartiallyWeightedCategoricalFocalLoss(alpha=0.25).to(device)
distance_loss_fn = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

liveplot = PlotLosses()

for epoch in range(num_epochs):
    logs = {}
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    train_loss = fit(model, train_loader, optimizer, semantic_loss_fn, distance_loss_fn)
    val_loss = validate(model, val_loader, semantic_loss_fn, distance_loss_fn)
    
    logs['log loss'] = train_loss
    logs['val_log loss'] = val_loss

    liveplot.update(logs)
    liveplot.draw()

    # Save the plot to a file
    plt.savefig(f'live_plot_epoch_{epoch+1}.png')

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), "SEDDModel.pth")

