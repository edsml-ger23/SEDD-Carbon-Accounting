#!/usr/bin/env python
# coding: utf-8

# ASPP_post.py

import bz2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import ASPP_utils as amu
import torch
import pandas as pd
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.patches as patches
from PIL import Image
import random
import os

root_dir = 'data/tiles/processed/'

def visualize_step(image, title, save_path, cmap='gray'):
    try:
        plt.figure(figsize=(6, 6))
        plt.title(title)
        
        if image.ndim == 3:
            if image.shape[0] == 3 or image.shape[0] == 1:
                image = image.transpose(1, 2, 0)
                print("Image was reshaped from (3, 4000, 4000) to (4000, 4000, 3)")
            elif image.shape[0] == 6:
                image = np.argmax(image, axis=0)
                print("Image had 6 channels, so it was converted to a single channel")
        
        elif image.ndim == 2:
            print("2D image detected, no reshaping necessary.")
        
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
        
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Failed to visualize image. Error: {e}")

def show_image_with_legend(image, title, save_path, cmap=None, norm=None, legend_info=None):
    try:
        if image.ndim == 3:
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
                print("Image was reshaped from (3, 4000, 4000) to (4000, 4000, 3)")
            elif image.shape[0] == 6:
                image = np.argmax(image, axis=0)
                print("Image had 6 channels, so it was converted to a single channel")

        elif image.ndim == 2:
            print("2D image detected, no reshaping necessary.")

        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")

        fig, ax = plt.subplots(figsize=(8, 6))
        if title:
            ax.set_title(title, fontsize=16)

        im = ax.imshow(image, cmap=cmap, norm=norm)
        ax.axis('off')

        plt.subplots_adjust(right=0.8)

        if legend_info:
            legend_patches = [patches.Patch(color=info['color'], label=info['label']) for info in legend_info]
            legend = ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Failed to visualize image with legend. Error: {e}")

species_values = [0, 1, 2, 3, 4, 5]
species_colors = ['purple', 'red', 'blue', 'green', 'yellow', 'orange']
species_names = ["Background", "Musacea", "Guaba", "Cacao", "Mango", "Otra Variedad"]
cmap = ListedColormap(species_colors)
norm = BoundaryNorm(species_values + [max(species_values) + 1], cmap.N)
legend_info = [{'color': color, 'label': name} for color, name in zip(species_colors, species_names)]

species_map_path = 'species_map_2_0.5.pkl'
with bz2.BZ2File(species_map_path, 'rb') as f:
    species_maps = pickle.load(f)

distance_map_path = 'distance_map_2_0.5.pkl'
with bz2.BZ2File(distance_map_path, 'rb') as f:
    distance_maps = pickle.load(f)

probability_map_path = 'probability_map_2_0.5.pkl'
with bz2.BZ2File(probability_map_path, 'rb') as f:
    probability_maps = pickle.load(f)

original_image_path = 'original_image_2_0.5.pkl'
with bz2.BZ2File(original_image_path, 'rb') as f:
    original_images = pickle.load(f)

species_ground_truth_path = 'species_map_true_2_0.5.pkl'
with bz2.BZ2File(species_ground_truth_path, 'rb') as f:
    species_ground_truth = pickle.load(f)

distance_map_truth_path = 'distance_map_true_2_0.5.pkl'
with bz2.BZ2File(distance_map_truth_path, 'rb') as f:
    distance_map_ground = pickle.load(f)

print("All maps loaded")

image_index = 0

visualize_step(original_images[image_index], 'Original Image', 'original_image_loaded.png')
show_image_with_legend(species_maps[image_index], 'Species Map', 'species_map_loaded.png', cmap=cmap, norm=norm, legend_info=legend_info)
visualize_step(distance_maps[image_index], 'Distance Map', 'distance_map_loaded.png')
visualize_step(probability_maps[image_index], 'Probability Map', 'probability_map_loaded.png')
visualize_step(species_ground_truth[image_index], 'Species Ground Truth', 'species_ground_truth_loaded.png')
visualize_step(distance_map_ground[image_index], 'Distance Ground Truth', 'distance_ground_truth_loaded.png')

print("All maps visualized")

selected_prob_map = probability_maps[0]
max_probabilities = np.max(selected_prob_map.transpose(1, 2, 0), axis=2)
overall_max_probability = np.max(max_probabilities)
max_location = np.unravel_index(np.argmax(max_probabilities, axis=None), max_probabilities.shape)
overall_max_probability_percentage = overall_max_probability * 100

print(f"Overall Maximum Probability: {overall_max_probability_percentage:.2f}%")
print(f"Location of Overall Maximum Probability: {max_location}")

plt.figure(figsize=(10, 10))
plt.imshow(max_probabilities, cmap='hot', interpolation='none')
plt.colorbar(label='Max Probability')
plt.scatter(max_location[1], max_location[0], color='blue', marker='o', s=100, label='Max Probability Location')
plt.legend()
plt.title('Maximum Probability for Each Pixel with Overall Max Location')
plt.savefig('probability_most_likely.png', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

def find_low_prob_pixels(probability_map): 
    max_probabilities = np.max(probability_map.transpose(1, 2, 0), axis=2)
    low_prob_pixels = np.where(max_probabilities < 0.2)
    return low_prob_pixels

for probability_map in probability_maps:
    low_prob_pixels = find_low_prob_pixels(probability_map)
    total_pixels = probability_map.shape[1] * probability_map.shape[2]
    low_prob_pixel_count = len(low_prob_pixels[0])
    percentage_low_prob_pixels = (low_prob_pixel_count / total_pixels) * 100
    print(f"Percentage of Pixels with Probability < 0.2: {percentage_low_prob_pixels:.2f}%")

def modify_species_map(species_map, low_prob_pixels, null_value=-1):
    modified_species_map = np.copy(species_map)
    for channel in range(modified_species_map.shape[0]):
        modified_species_map[channel, low_prob_pixels[0], low_prob_pixels[1]] = null_value
    return modified_species_map

modified_species_map_list = []
for probability_map, species_map in zip(probability_maps, species_maps):
    low_prob_pixels = find_low_prob_pixels(probability_map)
    modified_species_map = modify_species_map(species_map, low_prob_pixels)
    modified_species_map_list.append(modified_species_map)

del species_maps
species_maps_pred = np.array(modified_species_map_list)
species_maps_pred = np.argmax(species_maps_pred, axis=1)

distance_maps_pred = np.array(distance_maps)
del distance_maps
targets = np.array(species_ground_truth)
del species_ground_truth
distance_map_truth = np.array(distance_map_ground)
del distance_map_ground

if targets.shape[1] == 1:
    targets = np.squeeze(targets, axis=1)

if distance_maps_pred.shape[1] == 1:
    distance_maps_pred = np.squeeze(distance_maps_pred, axis=1)

print(f"targets shape: {targets.shape}")
print(f"distance_maps shape: {distance_map_truth.shape}")
print(f"species_maps_pred shape: {species_maps_pred.shape}")
print(f"distance_maps_pred shape: {distance_maps_pred.shape}")
print("All maps are now in the correct shape")

def calculate_f1_score(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
    return f1

average = 0
for i in range(len(targets)):
    current = calculate_f1_score(targets[i], species_maps_pred[i])
    print(current)
    average += current
average /= len(targets)
print("Average F1 Score:", average)

deepforest_results = pd.read_csv('iou_results_deepforest_ASPP.csv')
deepforest_results_grouped = deepforest_results.groupby('image_path')

def mark_boxes(image_path, root_dir, bounding_boxes):
    full_image_path = os.path.join(root_dir, image_path)
    image = cv2.imread(full_image_path)
    if image is None:
        print(f"Image at {image_path} could not be loaded.")
        return None
    
    white_background = np.ones_like(image) * 255
    
    for bbox in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(white_background, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), -1)
    
    return white_background

def visualize_boxes(image_path, df, root_dir, save_path):
    bounding_boxes = df[['pred_box_xmin', 'pred_box_ymin', 'pred_box_xmax', 'pred_box_ymax']].values.tolist()
    annotated_image = mark_boxes(image_path, root_dir, bounding_boxes)

    if annotated_image is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title("Annotated Image with Black Boxes on White Background")
        plt.show()
        plt.close()
        
        annotated_image = Image.fromarray(annotated_image)
        annotated_image.save(save_path)
    else:
        print("Failed to annotate image.")
    
    return annotated_image

binary_image_list = []
for image_path, group in deepforest_results_grouped:
    save_path = f'binary_image_{os.path.splitext(os.path.basename(image_path))[0]}.png'
    binary_image_list.append(visualize_boxes(image_path, group, root_dir, save_path))

bw_binary = []
for i in range(len(binary_image_list)):
    try:
        image_array = np.array(binary_image_list[i])
        bw_binary.append(cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY))
    except Exception as e:
        print(f"Failed to convert image at index {i} to grayscale. Error: {e}")

print("Binary images converted to black and white")

def select_dominant_species_normalized(colored_image, df_boxes, species_proportions, bias_factor=1, return_type='image'):
    labels = np.unique(colored_image)
    inverse_proportions = {label: (1.0 / prop) ** bias_factor for label, prop in species_proportions.items() if prop > 0}
    min_inverse = min(inverse_proportions.values(), default=1)
    inverse_proportions = {label: inverse_proportions.get(label, min_inverse / 10) for label in labels}

    result_image = np.zeros_like(colored_image, dtype=int)
    results_list = []

    for _, box in df_boxes.iterrows():
        xmin, ymin, xmax, ymax = map(int, [box['pred_box_xmin'], box['pred_box_ymin'], box['pred_box_xmax'], box['pred_box_ymax']])
        region = colored_image[ymin:ymax, xmin:xmax]
        if region.size == 0:
            dominant_species = np.nan
            if return_type == 'dataframe':
                results_list.append({**box, 'predicted_species': dominant_species})
            continue

        species_in_box, counts = np.unique(region, return_counts=True)
        if species_in_box.size == 1 and species_in_box[0] == 0:
            dominant_species_after = 5
            result_image[ymin:ymax, xmin:xmax] = dominant_species_after
            if return_type == 'dataframe':
                results_list.append({
                    'true_box_xmin': box['true_box_xmin'],
                    'true_box_ymin': box['true_box_ymin'],
                    'true_box_xmax': box['true_box_xmax'],
                    'true_box_ymax': box['true_box_ymax'],
                    'image_path': box['image_path'],
                    'name': box['name'],
                    'AGB': box['AGB'],
                    'carbon': box['carbon'],
                    'diameter': box['diameter'],
                    'pred_box_xmin': xmin,
                    'pred_box_ymin': ymin,
                    'pred_box_xmax': xmax,
                    'pred_box_ymax': ymax,
                    'iou': box['iou'],
                    'predicted_species': dominant_species_after
                })
            continue

        if 0 in species_in_box:
            zero_index = np.where(species_in_box == 0)[0][0]
            species_in_box = np.delete(species_in_box, zero_index)
            counts = np.delete(counts, zero_index)

        if species_in_box.size == 0:
            dominant_species = np.nan
            if return_type == 'dataframe':
                results_list.append({**box, 'predicted_species': dominant_species})
            continue

        normalized_counts = [count * inverse_proportions.get(species, min_inverse / 10) for species, count in zip(species_in_box, counts)]
        dominant_species_after = species_in_box[np.argmax(normalized_counts)]
        result_image[ymin:ymax, xmin:xmax] = dominant_species_after

        if return_type == 'dataframe':
            results_list.append({
                'true_box_xmin': box['true_box_xmin'],
                'true_box_ymin': box['true_box_ymin'],
                'true_box_xmax': box['true_box_xmax'],
                'true_box_ymax': box['true_box_ymax'],
                'image_path': box['image_path'],
                'name': box['name'],
                'AGB': box['AGB'],
                'carbon': box['carbon'],
                'diameter': box['diameter'],
                'pred_box_xmin': xmin,
                'pred_box_ymin': ymin,
                'pred_box_xmax': xmax,
                'pred_box_ymax': ymax,
                'iou': box['iou'],
                'predicted_species': dominant_species_after
            })

    if return_type == 'dataframe':
        return pd.DataFrame(results_list)
    else:
        return result_image

def select_dominant_species(colored_image, df_boxes, return_type='image'):
    result_image = np.zeros_like(colored_image, dtype=int)
    results_list = []

    for _, box in df_boxes.iterrows():
        xmin, ymin, xmax, ymax = map(int, [box['pred_box_xmin'], box['pred_box_ymin'], box['pred_box_xmax'], box['pred_box_ymax']])
        region = colored_image[ymin:ymax, xmin:xmax]
        if region.size == 0:
            continue

        species_in_box, counts = np.unique(region, return_counts=True)
        if 0 in species_in_box:
            zero_index = np.where(species_in_box == 0)[0][0]
            species_in_box = np.delete(species_in_box, zero_index)
            counts = np.delete(counts, zero_index)

        if species_in_box.size == 0:
            continue
        
        dominant_species = species_in_box[np.argmax(counts)]
        result_image[ymin:ymax, xmin:xmax] = dominant_species

        if return_type == 'dataframe':
            results_list.append({
                'true_box_xmin': box['true_box_xmin'],
                'true_box_ymin': box['true_box_ymin'],
                'true_box_xmax': box['true_box_xmax'],
                'true_box_ymax': box['true_box_ymax'],
                'image_path': box['image_path'],
                'name': box['name'],
                'AGB': box['AGB'],
                'carbon': box['carbon'],
                'diameter': box['diameter'],
                'pred_box_xmin': xmin,
                'pred_box_ymin': ymin,
                'pred_box_xmax': xmax,
                'pred_box_ymax': ymax,
                'iou': box['iou'],
                'predicted_species': dominant_species
            })
            return pd.DataFrame(results_list)
    else:
        return result_image

species_proportions = {
    1: 0.565334,  # Musacea
    2: 0.200999,  # Guaba
    3: 0.089547,  # Cacao
    5: 0.112221,  # Otra variedad
    4: 0.031899,  # Mango
}

def process_images(deep_forest_df, species_map_pred):
    results = []
    results_normalized = []

    for i in range(len(species_map_pred)):
        group_keys = list(deep_forest_df.groups.keys())
        curr_key = group_keys[i]
        df = deep_forest_df.get_group(curr_key)
        species_tester = species_map_pred[i]
        
        result_image = select_dominant_species(species_tester, df)
        result_image_normalized = select_dominant_species_normalized(species_tester, df, species_proportions)
        
        results.append(result_image)
        results_normalized.append(result_image_normalized)

    return results, results_normalized

results, results_normalized = process_images(deepforest_results_grouped, species_maps_pred)

'''____________________________________________________________________________________________________________________'''
#save and visualize the results

# Define the directory name
output_dir = 'example_species_maps'

# Check if the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")
else:
    print(f"Directory '{output_dir}' already exists.")

# Number of random images to select
num_images = 12

# Randomly select indices from the available images
random_indices = random.sample(range(len(results)), num_images)

# Loop through the selected indices and save the images in the specified directory
for i, idx in enumerate(random_indices):
    show_image_with_legend(results[idx], title=f'Dominant Species Map {i+1}', save_path=f'{output_dir}/example_species_map_{i+1}.png', cmap=cmap, norm=norm, legend_info=legend_info)
    show_image_with_legend(results_normalized[idx], title=f'Normalized Dominant Species Map {i+1}', save_path=f'{output_dir}/example_normalized_map_{i+1}.png', cmap=cmap, norm=norm, legend_info=legend_info)
    show_image_with_legend(targets[idx], title=f'True Species Map {i+1}', save_path=f'{output_dir}/example_target_map_{i+1}.png', cmap=cmap, norm=norm, legend_info=legend_info)

'''____________________________________________________________________________________________________________________'''
# Calculate the F1 scores of the results

f1_scores = []
f1_scores_normalized = []

for i in range(len(targets)):
    non_normal = calculate_f1_score(targets[i], results[i])
    print(f"F1 Score {i}: {non_normal:.4f}")
    f1_scores.append(non_normal)
    normal = calculate_f1_score(targets[i], results_normalized[i])
    print(f"F1 Score Normalized {i}: {normal:.4f}")
    f1_scores_normalized.append(normal)

print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
print(f"Average F1 Score (Normalized): {np.mean(f1_scores_normalized):.4f}")

df = pd.read_csv('test_data_ASPP.csv')

df_original_renamed = df.rename(columns={
    'xmin': 'true_box_xmin',
    'xmax': 'true_box_xmax',
    'ymin': 'true_box_ymin',
    'ymax': 'true_box_ymax',
    'img_path': 'image_path'
})

selected_columns = ['true_box_xmin', 'true_box_xmax', 'true_box_ymin', 'true_box_ymax', 'image_path', 'name', 'AGB', 'carbon', 'diameter']
df_original_selected = df_original_renamed[selected_columns]

merged_df = pd.merge(df_original_selected, deepforest_results, on=['true_box_xmin', 'true_box_ymin', 'true_box_xmax', 'true_box_ymax', 'image_path'])
grouped_merged = merged_df.groupby('image_path')

all_results = []

for species_map, (image_path, group_df) in zip(species_maps_pred, grouped_merged):
    result_df = select_dominant_species_normalized(species_map, group_df, species_proportions, return_type='dataframe')
    all_results.append(result_df)

final_df = pd.concat(all_results, ignore_index=True)

species_dict = {
    1: 'Musacea',
    2: 'Guaba',
    3: 'Cacao',
    4: 'Mango',
    5: 'Otra Variedad'
}

final_df['predicted_species'] = final_df['predicted_species'].map(species_dict)

final_df.to_csv('final_df_ASPP.csv', index=False)
