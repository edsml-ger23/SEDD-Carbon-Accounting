#post_utils.py

#necessary imports
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
import post_utils as pu

def visualize_step(image, title, save_path = None, cmap='gray'):
    try:
        plt.figure(figsize=(6, 6))
        plt.title(title)
        
        if image.ndim == 3:
            if image.shape[0] == 3 or image.shape[0] == 1:
                image = image.transpose(1, 2, 0)
            elif image.shape[0] == 6:
                image = np.argmax(image, axis=0)
        
        elif image.ndim == 2:
            pass
        
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
        
        plt.imshow(image, cmap=cmap)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()

    except Exception as e:
        print(f"Failed to visualize image. Error: {e}")


def show_image_with_legend(image, title, save_path = None, cmap=None, norm=None, legend_info=None):
    try:
        if image.ndim == 3:
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            elif image.shape[0] == 6:
                image = np.argmax(image, axis=0)

        elif image.ndim == 2:
            pass

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
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()

    except Exception as e:
        print(f"Failed to visualize image with legend. Error: {e}")

def set_visuals(): 
    species_values = [0, 1, 2, 3, 4, 5]
    species_colors = ['purple', 'red', 'blue', 'green', 'yellow', 'orange']
    species_names = ["Background", "Musacea", "Guaba", "Cacao", "Mango", "Otra Variedad"]
    cmap = ListedColormap(species_colors)
    norm = BoundaryNorm(species_values + [max(species_values) + 1], cmap.N)
    legend_info = [{'color': color, 'label': name} for color, name in zip(species_colors, species_names)]

    return cmap, norm, legend_info

def load_map(path): 
    with bz2.BZ2File(path, 'rb') as f:
        title = pickle.load(f)
    return title

def select_and_visualize_prob(prob_map): 
    max_probabilities = np.max(prob_map.transpose(1, 2, 0), axis=2)

    # Find the overall maximum probability
    overall_max_probability = np.max(max_probabilities)

    # Find the location of the overall maximum probability
    max_location = np.unravel_index(np.argmax(max_probabilities, axis=None), max_probabilities.shape)

    # Convert the overall maximum probability to percentage
    overall_max_probability_percentage = overall_max_probability * 100

    # Print the overall maximum probability percentage and its location
    print(f"Overall Maximum Probability: {overall_max_probability_percentage:.2f}%")
    print(f"Location of Overall Maximum Probability: {max_location}")

    # Optionally, visualize the location of the overall maximum probability on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(max_probabilities, cmap='hot', interpolation='none')
    plt.colorbar(label='Max Probability')
    plt.scatter(max_location[1], max_location[0], color='blue', marker='o', s=100, label='Max Probability Location')
    plt.legend()
    plt.title('Maximum Probability for Each Pixel with Overall Max Location')
    plt.show()

def find_low_prob_pixels(probability_map): 
    # Find the pixels with probability less than 0.2
    max_probabilities = np.max(probability_map.transpose(1, 2, 0), axis=2)
    low_prob_pixels = np.where(max_probabilities < 0.2)
    return low_prob_pixels

def print_low_prob(probability_maps): 
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

def apply_modification(probability_maps, species_maps): 
    modified_species_map_list = []
    for probability_map, species_map in zip(probability_maps, species_maps):
        low_prob_pixels = find_low_prob_pixels(probability_map)
        modified_species_map = modify_species_map(species_map, low_prob_pixels)
        modified_species_map_list.append(modified_species_map)
    return modified_species_map_list

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score between the true and predicted species maps.
    
    Parameters:
    - y_true: The true species map.
    - y_pred: The predicted species map.
    
    Returns:
    - f1_score: The F1 score.
    """
    # Flatten the true and predicted species maps
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate the F1 score
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
    return f1

def calculate_average_f1(targets, species_maps_pred): 
    average = 0
    for i in range(len(targets)):
        current = calculate_f1_score(targets[i], species_maps_pred[i])
        print(current)
        average += current
    average /= len(targets)
    print("Average F1 Score:", average)

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

def visualize_boxes(image_path, df, root_dir, show = True, save_path = None):
    bounding_boxes = df[['pred_box_xmin', 'pred_box_ymin', 'pred_box_xmax', 'pred_box_ymax']].values.tolist()
    annotated_image = mark_boxes(image_path, root_dir, bounding_boxes)

    if show:
        if annotated_image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_image)
            plt.axis('off')
            plt.title("Annotated Image with Black Boxes on White Background")
            plt.show()

            if save_path:
                plt.close()
                
                annotated_image = Image.fromarray(annotated_image)
                annotated_image.save(save_path)
        else:
            print("Failed to annotate image.")
    
    return annotated_image

def create_binary_deepforest(deepforest_results_grouped, root_dir, show = True):
    binary_image_list = []
    for image_path, group in deepforest_results_grouped:
        binary_box = visualize_boxes(image_path, group, root_dir, show = False)
        bw_binary = cv2.cvtColor(binary_box, cv2.COLOR_BGR2GRAY)
        binary_image_list.append(bw_binary)

    return binary_image_list

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

def species_and_deepforest_df(species_maps_pred, grouped_merged):
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

    return final_df

def change_column_names_back(df): 
    df.rename(columns={'true_box_xmin': 'xmin', 'true_box_ymin': 'ymin', 'true_box_xmax': 'xmax', 'true_box_ymax': 'ymax', 'image_path': 'img_path'}, inplace=True)
    return df

def change_column_names(df): 
    df.rename(columns={'xmin': 'true_box_xmin', 'ymin': 'true_box_ymin', 'xmax': 'true_box_xmax', 'ymax': 'true_box_ymax', 'img_path': 'image_path'}, inplace=True)
    return df

def calculate_AGB_averages(df, species_df): 
    print("Including all species")
    print("=====================================")
    print(f"Real AGB Average: {df['AGB'].mean()}")
    print(f"Predicted AGB Average: {df['predicted_AGB'].mean()}")
    print(f"AGB Difference Average: {df['AGB_difference'].mean()}")

    print(" ")

    print("Excluding species that don't match their predicted species")
    print("=====================================")
    print(f"Real AGB Average: {species_df['AGB'].mean()}")
    print(f"Predicted AGB Average: {species_df['predicted_AGB'].mean()}")
    print(f"AGB Difference Average: {species_df['AGB_difference'].mean()}")

def averages_by_image(df, species_df): 
    print("Including all species")
    print("=====================================")
    print(df.groupby('img_path')['AGB_difference'].mean())

    print(" ")

    print("Excluding species that don't match their predicted species")
    print("=====================================")
    print(species_df.groupby('img_path')['AGB_difference'].mean())

def sum_tables(df, species_df): 
    AGB_sum = df.groupby('img_path')['AGB'].sum()
    pred_sum = df.groupby('img_path')['predicted_AGB'].sum()

    #display side by side
    all = pd.concat([AGB_sum, pred_sum], axis=1)
    print("Including all species")
    print(all)

    print(" ")

    AGB_sum = species_df.groupby('img_path')['AGB'].sum()
    pred_sum = species_df.groupby('img_path')['predicted_AGB'].sum()

    #display side by side
    species = pd.concat([AGB_sum, pred_sum], axis=1)
    print("Excluding species that don't match their predicted species")
    print(species)

def carbon_tables(df, species_df): 
    AGB_sum = df.groupby('img_path')['AGB'].sum() *0.5
    pred_sum = df.groupby('img_path')['predicted_AGB'].sum() *0.5

    #display side by side
    all = pd.concat([AGB_sum, pred_sum], axis=1)
    print("Including all species")
    print(all)

    print(" ")

    AGB_sum = species_df.groupby('img_path')['AGB'].sum() * 0.5
    pred_sum = species_df.groupby('img_path')['predicted_AGB'].sum() * 0.5

    #display side by side
    species = pd.concat([AGB_sum, pred_sum], axis=1)
    print("Excluding species that don't match their predicted species")
    print(species)
    

