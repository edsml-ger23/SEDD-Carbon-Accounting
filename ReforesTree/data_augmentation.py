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
from scipy.ndimage import distance_transform_edt, gaussian_filter
from livelossplot import PlotLosses
import os
import time
import torchvision.transforms.functional as F

# Loading in the data
df = pd.read_csv('final_dataset_filtered.csv')
print('Data has been loaded')

# Generating Distance Maps
def generate_distance_map(image_shape, bbox):
    #create distance map with zeros
    distance_map = np.zeros(image_shape, dtype=np.float32)

    #extract bounding box coordinates 
    xmin, ymin, xmax, ymax = bbox.int().tolist()

    #create mask with zeros; bounding box set to 1
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 1

    #compute distance transform
    distance_map = distance_transform_edt(mask == 0)

    #apply Gaussian smoothing
    distance_map = gaussian_filter(distance_map, sigma=2)
    
    # Normalize the distance map to the range [0, 1]
    distance_map = distance_map / distance_map.max()  # Normalize distances to [0, 1]
    
    return distance_map

def generate_species_and_ID_map(image_shape, bbox, species_label, ID_label):
    # Initialize maps with zeros
    species_map = np.zeros(image_shape, dtype=np.uint8)
    ID_map = np.zeros(image_shape, dtype=np.uint8)

    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox.int().tolist()

    # Create masks within the bounding box
    species_map[ymin:ymax, xmin:xmax] = species_label 
    ID_map[ymin:ymax, xmin:xmax] = ID_label

    return species_map, ID_map

def how_much_tree(image): 
    #count the number of pixels in the image that are not 0
    pixel_num = np.count_nonzero(image)
    
    #return the percentage of pixels that are not 0
    return pixel_num/(image.shape[0]*image.shape[1])

#times included for debugging, but currently not printing 

class TreeCrownDataset(Dataset):
    def __init__(self, dataframe, root_dir, split, num_crops=8, transform=None, val_size=0.15, test_size=0.15, random_state=42):
        self.root_dir = root_dir
        self.transform = transform
        self.num_crops = num_crops #the set number at 8 is for a batch size of 2, generating approximately 20000 crops per epoch

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
        image = np.array(image)

        # Get metadata
        species = row['name']
        species_label = 1 if species == 'Musacea' else 2 if species == 'Guaba' else 3 if species == 'Cacao' else 4 if species == 'Mango' else 5
        ID_label = row['unique_id']
        bbox = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32)

        height, width, _ = image.shape

        # Generate maps
        species_map, ID_map = generate_species_and_ID_map((height, width), bbox, species_label, ID_label)
        distance_map = generate_distance_map((height, width), bbox)

        crop_size = 224

        # Initialize lists to hold multiple crops
        crop_imgs = []
        crop_species_maps = []
        crop_dist_maps = []
        crop_ID_maps = []

        count = 0

        while count < self.num_crops:
            # Random cropping logic
            x = random.randint(0, width - crop_size)
            y = random.randint(0, height - crop_size)

            crop_img = image[y:y+crop_size, x:x+crop_size]
            crop_species = species_map[y:y+crop_size, x:x+crop_size]
            crop_dist = distance_map[y:y+crop_size, x:x+crop_size]
            crop_ID = ID_map[y:y+crop_size, x:x+crop_size]

            crop_img = Image.fromarray(crop_img)
            crop_species = Image.fromarray(crop_species.astype(np.uint8))
            crop_dist = Image.fromarray(crop_dist.astype(np.float32))
            crop_ID = Image.fromarray(crop_ID.astype(np.uint8))

            if self.transform:
                crop_img = self.transform(crop_img)
                crop_species = self.transform(crop_species)
                crop_dist = self.transform(crop_dist)
                crop_ID = self.transform(crop_ID)

            # Check if the crop contains at least 10% of the tree crown
            if how_much_tree(crop_species) > 0.1:
                count += 1
                crop_imgs.append(crop_img)
                crop_species_maps.append(crop_species)
                crop_dist_maps.append(crop_dist)
                crop_ID_maps.append(crop_ID)
            else:
                continue

        # Stack lists into tensors
        crop_imgs = torch.stack(crop_imgs)
        crop_species_maps = torch.stack(crop_species_maps)
        crop_dist_maps = torch.stack(crop_dist_maps)
        crop_ID_maps = torch.stack(crop_ID_maps)

        return crop_imgs, crop_species_maps, crop_dist_maps, crop_ID_maps

def random_rotation(image):
    rotations = [0, 90, 180, 270]
    angle = random.choice(rotations)
    return transforms.functional.rotate(image, angle)

# Define the transform pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Lambda(lambda img: random_rotation(img)),
    transforms.ToTensor()
])

root_dir = 'data/tiles/processed'

train_dataset = TreeCrownDataset(dataframe=df, root_dir=root_dir, split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

val_dataset = TreeCrownDataset(dataframe=df, root_dir=root_dir, split='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)

test_dataset = TreeCrownDataset(dataframe=df, root_dir=root_dir, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

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

device = set_device("cuda")

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

        semantic_output = torch.nn.functional.interpolate(semantic_output, size=(224, 224), mode='bilinear', align_corners=True)
        distance_output = torch.nn.functional.interpolate(distance_output, size=(224, 224), mode='bilinear', align_corners=True)

        return semantic_output, distance_output

def final_loss(semantic_loss, distance_loss):
    return semantic_loss + distance_loss

def final_loss(semantic_loss, distance_loss):
    return semantic_loss + distance_loss

def fit(model, dataloader, optimizer, semantic_loss_fn, distance_loss_fn):
    model.train()
    running_loss = 0.0

    for images, targets, distance_maps, _ in dataloader:

        #reshaping from 5D tensor (batches and crops) to 4D tensor (batches*crops)
        images = images.view(-1, images.size(2), images.size(3), images.size(4))  # [B*ncrops, C, H, W]
        targets = targets.view(-1, targets.size(3), targets.size(4))  # [B*ncrops, H, W]
        distance_maps = distance_maps.view(-1, distance_maps.size(2), distance_maps.size(3), distance_maps.size(4))  # [B*ncrops, H, W]

        images = images.to(device)
        targets = targets.to(device)
        distance_maps = distance_maps.to(device)

        optimizer.zero_grad()
        semantic_outputs, distance_outputs = model(images)
        semantic_outputs_scaled = semantic_outputs * 255

        # Ensure target tensors are cast to appropriate types
        semantic_loss = semantic_loss_fn(semantic_outputs_scaled, targets.to(torch.long))  
        distance_loss = distance_loss_fn(distance_outputs, distance_maps.to(torch.float32))

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
        for images, targets, distance_maps, _ in dataloader:
            #again reshaping from 5D tensor to 4D tensor
            images = images.view(-1, images.size(2), images.size(3), images.size(4))  # [B*ncrops, C, H, W]
            targets = targets.view(-1, targets.size(3), targets.size(4))  # [B*ncrops, H, W]
            distance_maps = distance_maps.view(-1, distance_maps.size(2), distance_maps.size(3), distance_maps.size(4))  # [B*ncrops, H, W]

            images = images.to(device)
            targets = targets.to(device)
            distance_maps = distance_maps.to(device, dtype=torch.float32)

            semantic_outputs, distance_outputs = model(images)
            semantic_outputs_scaled = semantic_outputs * 255

            semantic_loss = semantic_loss_fn(semantic_outputs_scaled, targets.to(torch.long))
            distance_loss = distance_loss_fn(distance_outputs, distance_maps.to(torch.float32))
            loss = final_loss(semantic_loss, distance_loss)

            running_loss += loss.item()

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss

# Training the Model
num_epochs = 20
model = SEDDModel(num_classes=5).to(device)
semantic_loss_fn = PartiallyWeightedCategoricalFocalLoss(alpha=0.25).to(device)
distance_loss_fn = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loss = fit(model, train_loader, optimizer, semantic_loss_fn, distance_loss_fn)
    val_loss = validate(model, val_loader, semantic_loss_fn, distance_loss_fn)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

torch.save(model.state_dict(), "DistanceAndSpecies.pth")


