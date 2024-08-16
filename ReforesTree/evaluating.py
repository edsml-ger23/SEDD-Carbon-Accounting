#!/usr/bin/env python
# coding: utf-8

# evaluating.py
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import time
import models_utils as mu

test_df = pd.read_csv('test_df_proper.csv')
root_dir = 'data/tiles/processed'

print("Loading test dataset...")

test_dataset = mu.TestDataset(dataframe=test_df, root_dir=root_dir, transform=None)
# Converts to Tensor in dataloader, hence transform is None
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)

print("Test dataset loaded.")

device = mu.set_device("cuda")

model = mu.SEDDModel(num_classes=6).to(device)
model.load_state_dict(torch.load('correct_loading_take2_20.pth'))

model.eval()

patch_size = (224, 224)  # Patch size

# Set to 50% overlap
stride = (112, 112)  # 50% overlap

start_time = time.time()
accuracy, precision, recall, f1, mse, all_original_images, all_species_maps, all_distance_maps, all_probability_maps, all_distance_maps_true, all_species_maps_true = mu.full_evaluation(model, test_loader, patch_size, stride, device)
print(f"Time taken for 50% overlap: {time.time() - start_time}")

mu.print_results_and_save(
    accuracy, precision, recall, f1, mse,
    all_original_images, all_species_maps,
    all_distance_maps, all_probability_maps,
    all_distance_maps_true, all_species_maps_true,
    0.5
)

