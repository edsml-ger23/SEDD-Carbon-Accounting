#!/usr/bin/env python
# coding: utf-8

#SEDD_model.py
import pandas as pd 
import model_utils as mu
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

'''__________________________'''

#establishing the df and root_dir for whole file 
root_dir = 'data/tiles/processed'
df = pd.read_csv('csv_files/final_dataset_filtered.csv') 

#combining all species outside the 4 main species into one category
df['name'] = df['name'].apply(lambda x: x if x in ['Musacea','Guaba','Cacao','Mango'] else 'Otra variedad')

'''__________________________'''
#Loading and Visualizing the Data

#Custom Transformations
transform = mu.RandomTransforms(
    horizontal_flip_prob=0.5,
    vertical_flip_prob=0.5,
    rotation_degrees=[0, 90, 180, 270]
)

#splitting and loading the data
train_df, test_df, val_df = mu.split_data_SEDD(df)

train_data = mu.TreeCrownDataset(train_df, root_dir, transform=transform)
val_data = mu.TreeCrownDataset(val_df, root_dir, transform=transform)
test_data = mu.TestDataset(test_df, root_dir, transform=None)

#save the test data for later evaluation
test_df.to_csv('csv_files/test_data_SEDD.csv', index=False)

#loading the data
train_loader = DataLoader(train_data, batch_size=8, shuffle=False)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

#making sure there are distinct classes
train_class_distribution = mu.analyze_class_distribution(train_loader)
print("Train class distribution:", train_class_distribution)

val_class_distribution = mu.analyze_class_distribution(val_loader)
print("Validation class distribution:", val_class_distribution)

test_class_distribution = mu.analyze_class_distribution(test_loader)
print("Test class distribution:", test_class_distribution)

#Visualizing the test data 
test_iter = iter(test_loader)
images, species_maps, dist_maps = next(test_iter)

cmap, norm, legend_info = mu.set_visuals()

mu.show_images(images, title="Randomly Cropped Image Example")
mu.show_images_with_legend(species_maps,title="Species Map Example", cmap=cmap, norm=norm, legend_patches=legend_info)
mu.show_images(dist_maps,title="Distance Map Example")

'''__________________________'''
#Training the Model

#setting the device
device = mu.set_device("cuda")

model = mu.SEDDModel(num_classes=6).to(device)
semantic_loss_fn = mu.PartiallyWeightedCategoricalFocalLoss(alpha=0.25).to(device)
distance_loss_fn = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 15

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Fit the model and calculate training loss
    train_loss = mu.fit(model, train_loader, optimizer, semantic_loss_fn, distance_loss_fn, device)

    # Validate the model and calculate validation loss
    val_loss = mu.validate(model, val_loader, semantic_loss_fn, distance_loss_fn, device)

    # Step the learning rate scheduler
    scheduler.step()

    # Print epoch results
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    #save the model every 5 epochs
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"SEDD_model_{epoch+1}.pth")

model = mu.SEDDModel(num_classes=6).to(device)
model.load_state_dict(torch.load('SEDD_model_15.pth'))

model.eval()

patch_size = (224, 224)  # Patch size
stride = (112, 112)  # 50% overlap

accuracy, precision, recall, f1, mse, all_original_images, all_species_maps, all_distance_maps, all_probability_maps, all_distance_maps_true, all_species_maps_true = mu.full_evaluation(model, test_loader, patch_size, stride, device)

mu.print_results_and_save(
    accuracy, precision, recall, f1, mse,
    all_original_images, all_species_maps,
    all_distance_maps, all_probability_maps,
    all_distance_maps_true, all_species_maps_true,
    0.5
)

#set to 30% overlap
stride = (67, 67)  # 30% overlap

accuracy, precision, recall, f1, mse, all_original_images, all_species_maps, all_distance_maps, all_probability_maps, all_distance_maps_true, all_species_maps_true = mu.full_evaluation(model, test_loader, patch_size, stride, device)

mu.print_results_and_save(
    accuracy, precision, recall, f1, mse,
    all_original_images, all_species_maps,
    all_distance_maps, all_probability_maps,
    all_distance_maps_true, all_species_maps_true,
    0.3
)

#set to 10% overlap
stride = (22, 22)  # 10% overlap

accuracy, precision, recall, f1, mse, all_original_images, all_species_maps, all_distance_maps, all_probability_maps, all_distance_maps_true, all_species_maps_true = mu.full_evaluation(model, test_loader, patch_size, stride, device)

mu.print_results_and_save(
    accuracy, precision, recall, f1, mse,
    all_original_images, all_species_maps,
    all_distance_maps, all_probability_maps,
    all_distance_maps_true, all_species_maps_true,
    0.1
)