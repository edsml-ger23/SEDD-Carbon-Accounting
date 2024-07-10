#models_utils.py

#necessary imports
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

    # Debugging: Print bounding box and labels
    print(f"BBox: {xmin, ymin, xmax, ymax}, Species Label: {species_label}, ID Label: {ID_label}")

    # Create masks within the bounding box
    species_map[ymin:ymax, xmin:xmax] = species_label 
    ID_map[ymin:ymax, xmin:xmax] = ID_label

    return species_map, ID_map

def how_much_tree(image): 
    #count the number of pixels in the image that are not 0
    pixel_num = np.count_nonzero(image)
    
    #return the percentage of pixels that are not 0
    return pixel_num/(image.shape[0]*image.shape[1])

def visualize_distance_map(distance_map_dir, img_path, image_dir):
    """
    Load and visualize a precomputed distance map along with the original image.

    Parameters:
    - distance_map_dir (str): Directory where distance maps are stored.
    - img_path (str): Path of the image used to generate the distance map.
    - image_dir (str): Directory where original images are stored.
    """
    # Generate the filename for the distance map
    distance_map_filename = f"{img_path.replace('/', '_').replace('.png', '.pkl.bz2')}"
    distance_map_path = os.path.join(distance_map_dir, distance_map_filename)
    
    # Load the distance map
    with bz2.BZ2File(distance_map_path, 'rb') as f:
        distance_map = pickle.load(f)
    
    # Load the original image
    full_img_path = os.path.join(image_dir, img_path)
    original_image = Image.open(full_img_path)
    
    # Plot the original image and distance map side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(distance_map, cmap='gray')
    axes[1].set_title('Distance Map')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], orientation='vertical')
    
    plt.show()

class TreeCrownDataset(Dataset):
    def __init__(self, dataframe, root_dir, num_crops = 8, transform=None, random_state=42):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.num_crops = num_crops #this is hard coded for a batch size of 2


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
        species_label = 1 if species == 'Musacea' else 2 if species == 'Guaba' else 3 if species == 'Cacao' else 4 if species == 'Mango' else 5 if species == 'Otra variedad' else 0
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
            
            crop_img = transforms.ToTensor()(crop_img)
            crop_species = torch.tensor(crop_species, dtype=torch.int64)
            crop_dist = torch.tensor(crop_dist, dtype=torch.float32)
            crop_ID = torch.tensor(crop_ID, dtype=torch.int64)

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
    
class TestDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load image
        img_path = os.path.join(self.root_dir, row['img_path'])
        image = Image.open(img_path).convert('RGB')

        # Get metadata
        species = row['name']
        species_label = 1 if species == 'Musacea' else 2 if species == 'Guaba' else 3 if species == 'Cacao' else 4 if species == 'Mango' else 5 if species == 'Otra variedad' else 0
        ID_label = row['unique_id']
        bbox = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32)

        # Transform image if a transform is provided
        if self.transform:
            image = self.transform(image)

        # Generate maps
        height, width = image.size(1), image.size(2)  # assuming image is a tensor of shape (C, H, W)
        species_map, ID_map = generate_species_and_ID_map((height, width), bbox, species_label, ID_label)
        distance_map = generate_distance_map((height, width), bbox)

        return image, species_map, distance_map, ID_map

def random_rotation(tensor):
    rotations = [0, 90, 180, 270]
    angle = random.choice(rotations)

    if angle == 0:
        return tensor
    elif angle == 90:
        return tensor.transpose(-2, -1).flip(-2)  # Equivalent to rotating 90 degrees
    elif angle == 180:
        return tensor.flip(-2).flip(-1)  # Equivalent to rotating 180 degrees
    elif angle == 270:
        return tensor.transpose(-2, -1).flip(-1)  # Equivalent to rotating 270 degrees

    return tensor

def show_images(images, nrow=2, ncol=4, title=None):
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
    if title:
        fig.suptitle(title, fontsize=16)
        
    for i, img in enumerate(images):
        if i >= nrow * ncol:
            break
        ax = axes[i // ncol, i % ncol]
        
        # Check if img has 4 dimensions (batch, C, H, W)
        if img.dim() == 4:
            img = img[0]  # Take the first image in the batch
        
        # Ensure img has 3 dimensions (C, H, W) before permute
        if img.dim() == 3:
            img = img.permute(1, 2, 0)  # Rearrange dimensions for matplotlib (C, H, W) -> (H, W, C)
        
        ax.imshow(img)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def analyze_class_distribution(dataloader):
    all_targets = []

    with torch.no_grad():
        for _, targets, _, _ in dataloader:
            all_targets.append(targets.cpu().numpy())

    # Flatten the array for analysis
    all_targets = np.concatenate(all_targets).ravel()
    
    # Compute the distribution of each class
    unique, counts = np.unique(all_targets, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    return class_distribution

def set_device(device="cpu", idx=0):
    """
    Set the device for PyTorch operations.

    Parameters:
    - device (str, optional): Device to use ("cpu" or "cuda"). Default: "cpu".
    - idx (int, optional): Index of the GPU device if using CUDA. Default: 0.

    Returns:
    - str: The selected device.
    """
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
        self.dropout = nn.Dropout(p=0.65)  # Add dropout layer with a rate of 0.65

    def forward(self, x):
        x = self.deeplab_head(x)
        x = self.dropout(x)  # Apply dropout
        return x
    
class DistanceMapDecoder(nn.Module):
    def __init__(self):
        super(DistanceMapDecoder, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.65)  # Add dropout layer with a rate of 0.65

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.dropout(x)  # Apply dropout
        x = nn.Sigmoid()(self.conv2(x))
        return x
    
class PartiallyWeightedCategoricalFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, beta=0.75):
        super(PartiallyWeightedCategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        weights = torch.where(target == 0, self.alpha, self.beta)
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

def fit(model, dataloader, optimizer, semantic_loss_fn, distance_loss_fn, device):
    model.train()
    running_loss = 0.0

    for images, targets, distance_maps, _ in dataloader:

        print("Images shape:", images.shape)
        print("Targets shape:", targets.shape)
        print("Distance maps shape:", distance_maps.shape)

        #reshaping from 5D tensor (batches and crops) to 4D tensor (batches*crops)
        images = images.view(-1, images.size(2), images.size(3), images.size(4))  # [B*ncrops, C, H, W]
        targets = targets.view(-1, targets.size(2), targets.size(3))  # [B*ncrops, H, W]
        distance_maps = distance_maps.view(-1, distance_maps.size(2), distance_maps.size(3))

        images = images.to(device)
        targets = targets.to(device)
        distance_maps = distance_maps.to(device)

        optimizer.zero_grad()
        semantic_outputs, distance_outputs = model(images)

        # Ensure target tensors are cast to appropriate types
        semantic_loss = semantic_loss_fn(semantic_outputs, targets.to(torch.long))  
        distance_loss = distance_loss_fn(distance_outputs, distance_maps.to(torch.float32))

        loss = final_loss(semantic_loss, distance_loss)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader.dataset)
    return train_loss

def validate(model, dataloader, semantic_loss_fn, distance_loss_fn, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets, distance_maps, _ in dataloader:
            #again reshaping from 5D tensor to 4D tensor
            images = images.view(-1, images.size(2), images.size(3), images.size(4))  # [B*ncrops, C, H, W]
            targets = targets.view(-1, targets.size(2), targets.size(3))  # [B*ncrops, H, W]
            distance_maps = distance_maps.view(-1, distance_maps.size(2), distance_maps.size(3))

            images = images.to(device)
            targets = targets.to(device)
            distance_maps = distance_maps.to(device, dtype=torch.float32)

            semantic_outputs, distance_outputs = model(images)

            semantic_loss = semantic_loss_fn(semantic_outputs, targets.to(torch.long))
            distance_loss = distance_loss_fn(distance_outputs, distance_maps.to(torch.float32))
            loss = final_loss(semantic_loss, distance_loss)

            running_loss += loss.item()

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss

def sliding_window_inference(model, image, patch_size, stride, device, num_classes=6):
    """
    Perform sliding window inference on the input image.

    Parameters:
    - model: Trained model for prediction.
    - image: Input image tensor of shape (C, H, W).
    - patch_size: Size of the patches to extract (height, width).
    - stride: Stride for the sliding window.
    - device: Device to perform computations on ('cpu' or 'cuda').

    Returns:
    - final_prediction: Combined prediction map for the entire image.
    """
    _, H, W = image.shape
    patches = []
    coords = []

    # Extract patches using sliding window
    for y in range(0, H - patch_size[0] + 1, stride[0]):
        for x in range(0, W - patch_size[1] + 1, stride[1]):
            patch = image[:, y:y + patch_size[0], x:x + patch_size[1]]
            patches.append(patch)
            coords.append((y, x))

    # Convert to tensor and perform model inference
    patches = torch.stack(patches).to(device)
    with torch.no_grad():
        semantic_preds, distance_preds = model(patches)

    # Initialize final prediction maps
    semantic_prediction_map = torch.zeros((num_classes, H, W)).to(device)
    distance_prediction_map = torch.zeros((1, H, W)).to(device)
    count_map = torch.zeros((H, W)).to(device)

    # Combine patch predictions into final prediction maps
    for (y, x), semantic_pred, distance_pred in zip(coords, semantic_preds, distance_preds):
        # Print shapes for debugging
        
        semantic_prediction_map[:, y:y + patch_size[0], x:x + patch_size[1]] += semantic_pred
        distance_prediction_map[:, y:y + patch_size[0], x:x + patch_size[1]] += distance_pred
        count_map[y:y + patch_size[0], x:x + patch_size[1]] += 1

    # Avoid division by zero
    count_map[count_map == 0] = 1  # To avoid division by zero
    count_map = count_map.unsqueeze(0).repeat(num_classes, 1, 1)
    semantic_prediction_map /= count_map
    distance_prediction_map /= count_map[0]

    return semantic_prediction_map, distance_prediction_map

def evaluate_sliding_window(model, dataloader, patch_size, stride, device):
    model.eval()
    all_targets = []
    all_predictions = []
    all_distances_true = []
    all_distances_pred = []
    all_species_maps = []
    all_distance_maps = []

    with torch.no_grad():
        for images, targets, distance_maps, _ in dataloader:
            for i in range(images.shape[0]):
                image = images[i].to(device)
                target = targets[i].to(device)
                distance_map = distance_maps[i].to(device)

                # Perform sliding window inference
                semantic_pred, distance_pred = sliding_window_inference(
                    model, image, patch_size, stride, device)

                # Convert predictions to final class labels and append to lists
                predictions = torch.argmax(semantic_pred, dim=0)
                all_targets.append(target.cpu().numpy().ravel())
                all_predictions.append(predictions.cpu().numpy().ravel())
                all_distances_true.append(distance_map.cpu().numpy().ravel())
                all_distances_pred.append(distance_pred.cpu().numpy().ravel())

                # Store the species classification maps and distance maps
                all_species_maps.append(semantic_pred.cpu().numpy())
                all_distance_maps.append(distance_pred.cpu().numpy())

    # Flatten arrays for metric computation
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_distances_true = np.concatenate(all_distances_true)
    all_distances_pred = np.concatenate(all_distances_pred)

    # Compute evaluation metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    mse = np.mean((all_distances_true - all_distances_pred) ** 2)

    return accuracy, precision, recall, f1, mse, all_species_maps, all_distance_maps

