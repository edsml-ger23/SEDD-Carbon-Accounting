"""
data_utils.py

This module contains utility functions for processing and enhancing images, 
handling bounding boxes, and performing various pre-processing and post-processing 
tasks for image analysis. It also includes functions for model-based above-ground 
biomass (AGB) prediction and visualization.

Key functionalities include:
- Drawing and annotating bounding boxes on images.
- Calculating image metrics such as brightness, contrast, and edge intensity.
- Applying various image enhancement techniques, including CLAHE, gamma correction, 
  histogram equalization, and sharpening.
- Managing datasets by removing images with excessive white areas, applying enhancements, 
  and handling overlapping bounding boxes.
- Post-processing tasks like visualizing probability maps, modifying species maps, 
  and calculating evaluation metrics.
- Functions for loading and using Generalized Additive Models (GAMs) to predict 
  above-ground biomass (AGB) based on species and diameter.

This module is designed to support image-based analysis and modeling tasks, 
particularly in the context of calculating carbon storage sequestration via 
ITC delineation and species identification.
"""

#this is a test image path to be used in the doctest
image_path = 'data/tiles/processed/Carlos Vera Arteaga RGB_0_0_0_4000_4000.png'

# Standard Library Imports
import os
import random
import bz2
import pickle

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from sklearn.metrics import f1_score
from PIL import Image
import cv2
import pandas as pd

"""_______________________________________________________________"""

"""
Pre-Processing Utilities

This module contains functions that facilitate the pre-processing of image data, 
particularly for tasks involving environmental and ecological analysis. These 
utilities are designed to enhance image quality, manage datasets, and prepare data 
for machine learning models. Key functionalities include:

- Image Enhancement: Tools to improve image quality through techniques such as 
  CLAHE (Contrast Limited Adaptive Histogram Equalization), gamma correction, 
  and sharpening filters. These enhancements help to improve the visibility and 
  contrast of features within images.
- Bounding Box Operations: Functions to draw bounding boxes on images, visualize 
  annotated images, and remove images with excessive white areas, ensuring that 
  only high-quality data is used in model training and evaluation.
- Metric Calculation: Utilities to calculate image metrics like brightness, contrast, 
  and edge intensity, providing quantitative measures of image quality.
- Dataset Management: Functions to apply enhancements to entire datasets, remove 
  images with significant white areas, and ensure that bounding boxes do not 
  overlap excessively, improving the quality and consistency of the dataset.
- Intersection Over Union (IoU): A function to compute the Intersection over Union 
  (IoU) for bounding boxes, which is crucial for evaluating the accuracy of object 
  detection models.
- Outlier and Overlap Handling: Tools to discard overlapping bounding boxes based 
  on species counts, helping to clean the dataset by removing redundant or 
  conflicting data points.

These utilities are essential for preparing image data, enhancing image quality, 
and managing datasets effectively, setting the stage for accurate and reliable 
machine learning model training and evaluation.
"""

def draw_bounding_boxes(image_path, bounding_boxes):
    '''
    Draws bounding boxes on an image and returns the annotated image.

    Parameters:
    - image_path (str): The path to the image file that needs to be annotated.
    - bounding_boxes (list of dict): A list of dictionaries, where each dictionary 
      represents a bounding box with the following keys:
        - 'xmin' (int): The x-coordinate of the top-left corner of the bounding box.
        - 'ymin' (int): The y-coordinate of the top-left corner of the bounding box.
        - 'xmax' (int): The x-coordinate of the bottom-right corner of the bounding box.
        - 'ymax' (int): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
    - np.ndarray or None: The annotated image with bounding boxes drawn. If the image 
      cannot be loaded, the function returns None and prints an error message.

    Examples:
    >>> image_path = image_path
    >>> bounding_boxes = [{"xmin": 100, "ymin": 150, "xmax": 200, "ymax": 250}]
    >>> result = draw_bounding_boxes(image_path, bounding_boxes)
    >>> isinstance(result, np.ndarray)  # Check that the result is a numpy array
    True

    >>> non_existent_image = 'non_existent_image.png'
    >>> result = draw_bounding_boxes(non_existent_image, bounding_boxes)
    Image at non_existent_image.png could not be loaded.
    >>> result is None  # Check that None is returned for a non-existent image
    True
    '''

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at {image_path} could not be loaded.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw each bounding box
    for bbox in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        cv2.rectangle(
            image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 10
        )

    return image


def calculate_brightness(img):
    '''Calculates the brightness of an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    
    Returns:
    - float: The brightness of the image.
    
    Examples:
    >>> # Example 1: Solid white image (brightness should be 255.0)
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> calculate_brightness(white_image)
    255.0

    >>> # Example 2: Solid black image (brightness should be 0.0)
    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> calculate_brightness(black_image)
    0.0

    >>> # Example 3: Solid gray image (brightness should be 128.0)
    >>> gray_image = np.ones((10, 10, 3), dtype=np.uint8) * 128
    >>> calculate_brightness(gray_image)
    128.0
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def calculate_contrast(img):
    '''Calculates the contrast of an image.

    Parameters:
    - img (np.ndarray): The input image as a NumPy array.

    Returns:
    - float: The contrast of the image.
    
    Examples:
    >>> # Example 1: Solid white image (contrast should be 0.0)
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> calculate_contrast(white_image)
    0.0

    >>> # Example 2: Solid black image (contrast should be 0.0)
    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> calculate_contrast(black_image)
    0.0

    >>> # Example 4: High contrast image (black and white stripes)
    >>> stripe_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> stripe_image[::2] = 255  # Set every other row to white
    >>> calculate_contrast(stripe_image)
    127.5
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def calculate_edge_intensity(img):
    '''Calculates the edge intensity of an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    
    Returns:
    - float: The edge intensity of the image.
    
    Examples:
    >>> # Example 1: Solid white image (no edges, so edge intensity should be 0.0)
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> calculate_edge_intensity(white_image)
    0.0

    >>> # Example 2: Solid black image (no edges, so edge intensity should be 0.0)
    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> calculate_edge_intensity(black_image)
    0.0

    >>> # Example 4: Image with a white square on a black background (should have edges)
    >>> square_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> square_image[2:8, 2:8] = 255  # White square in the middle
    >>> calculate_edge_intensity(square_image) > 0
    True
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges) / 255


def calculate_metrics(img):
    '''Calculates the brightness, contrast, and edge intensity of an image.

    Parameters:
    - img (np.ndarray): The input image as a NumPy array.

    Returns:
    - dict: A dictionary containing the calculated metrics for the image.
    
    Examples:
    >>> # Example 1: Solid white image
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> calculate_metrics(white_image)
    {'brightness': 255.0, 'contrast': 0.0, 'edge_intensity': 0.0}

    >>> # Example 2: Solid black image
    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> calculate_metrics(black_image)
    {'brightness': 0.0, 'contrast': 0.0, 'edge_intensity': 0.0}

    >>> # Example 3: Solid gray image
    >>> gray_image = np.ones((10, 10, 3), dtype=np.uint8) * 128
    >>> calculate_metrics(gray_image)
    {'brightness': 128.0, 'contrast': 0.0, 'edge_intensity': 0.0}

    >>> # Example 4: Image with a white square on a black background
    >>> square_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> square_image[2:8, 2:8] = 255  # White square in the middle
    >>> metrics = calculate_metrics(square_image)
    >>> metrics['brightness'] > 0
    True
    >>> metrics['contrast'] > 0
    True
    >>> metrics['edge_intensity'] > 0
    True
    '''
    brightness = calculate_brightness(img)
    contrast = calculate_contrast(img)
    edge_intensity = calculate_edge_intensity(img)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_intensity": edge_intensity,
    }


def clahe(img):
    '''Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    
    Returns:
    - np.ndarray: The CLAHE-enhanced image.
    
    Examples:
    >>> # Example 1: Solid white image (CLAHE should have no effect)
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> enhanced_white_image = clahe(white_image)
    >>> np.array_equal(white_image, enhanced_white_image)
    True

    >>> # Example 2: Image with gradient (CLAHE should enhance contrast)
    >>> gradient_image = np.tile(np.arange(10, dtype=np.uint8).reshape(10, 1), (1, 10))
    >>> gradient_image = np.stack([gradient_image]*3, axis=-1)
    >>> enhanced_gradient_image = clahe(gradient_image)
    >>> not np.array_equal(gradient_image, enhanced_gradient_image)
    True
    '''
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    # Convert LAB image back to color (RGB)
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


# Display the original and enhanced image
def display_images(
    img, enhanced_img, title, baseline_metrics, original_title="Original Image"
):
    '''Displays the original and enhanced images side by side, along with the calculated metrics.

    Parameters:
    - img (np.ndarray): The original image as a NumPy array.
    - enhanced_img (np.ndarray): The enhanced image as a NumPy array.
    - title (str): The title for the enhanced image.
    - baseline_metrics (dict): The calculated metrics for the original image.
    - original_title (str): The title for the original image (default: 'Original Image').

    Returns:
    - None

    Since this is just a utility function for visualization, it does not need any tests.
    '''
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(original_title)
    #remove axis ticks 
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))

    #remove axis ticks 
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # also print the metrics for the enhanced image
    print("Original Metrics:")
    print(baseline_metrics)
    enhanced_metrics = calculate_metrics(enhanced_img)
    print("Enhanced Metrics:")
    print(enhanced_metrics)


def equalize_histogram(img):
    '''Applies histogram equalization to an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    
    Returns:
    - np.ndarray: The histogram-equalized image.
    
    Examples:
    >>> # Example 1: Solid white image (histogram equalization should have no effect)
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> equalized_white_image = equalize_histogram(white_image)
    >>> np.array_equal(white_image, equalized_white_image)
    True

    >>> # Example 2: Solid black image (histogram equalization should have no effect)
    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> equalized_black_image = equalize_histogram(black_image)
    >>> np.array_equal(black_image, equalized_black_image)
    True

    >>> # Example 3: Image with gradient (histogram equalization should alter the image)
    >>> gradient_image = np.tile(np.arange(10, dtype=np.uint8).reshape(10, 1), (1, 10))
    >>> gradient_image = np.stack([gradient_image]*3, axis=-1)
    >>> equalized_gradient_image = equalize_histogram(gradient_image)
    >>> not np.array_equal(gradient_image, equalized_gradient_image)
    True
    '''
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    return img_eq


def gamma_correction(img, gamma=1.0):
    '''Applies gamma correction to an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    - gamma (float): The gamma value for the correction (default: 1.0).
    
    Returns:
    - np.ndarray: The gamma-corrected image.

    Examples:
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> corrected_white_image = gamma_correction(white_image, gamma=2.0)
    >>> np.array_equal(white_image, corrected_white_image)
    True

    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> corrected_black_image = gamma_correction(black_image, gamma=0.5)
    >>> np.array_equal(black_image, corrected_black_image)
    True
    '''
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    corrected_img = cv2.LUT(img, table)

    return corrected_img


def linear_contrast_stretching(img, alpha=1.0, beta=0):
    '''Applies linear contrast stretching to an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    - alpha (float): The contrast scaling factor (default: 1.0).
    - beta (float): The brightness scaling factor (default: 0).
    
    Returns:
    - np.ndarray: The contrast-stretched image.
    
    Since this is basically just a wrapper for a built-in OpenCV function, we don't need to test it.'''
    stretched_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return stretched_img


def unsharp_mask(img, sigma=1.0, strength=1.5):
    '''Applies unsharp masking to an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    - sigma (float): The standard deviation for the Gaussian blur (default: 1.0).
    - strength (float): The strength of the sharpening effect (default: 1.5).
    
    Returns:
    - np.ndarray: The sharpened image.

    Examples:
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> sharpened_white_image = unsharp_mask(white_image)
    >>> np.array_equal(white_image, sharpened_white_image)
    True

    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> sharpened_black_image = unsharp_mask(black_image)
    >>> np.array_equal(black_image, sharpened_black_image)
    True
    '''
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    unsharp = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

    return unsharp


def laplacian_sharpening(img):
    '''Applies Laplacian sharpening to an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    
    Returns:
    - np.ndarray: The sharpened image.

    Examples:
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> sharpened_white_image = laplacian_sharpening(white_image)
    >>> np.array_equal(white_image, sharpened_white_image)
    True

    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> sharpened_black_image = laplacian_sharpening(black_image)
    >>> np.array_equal(black_image, sharpened_black_image)
    True
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert back to BGR
    laplacian = cv2.convertScaleAbs(laplacian)
    laplacian_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    sharpened = cv2.addWeighted(img, 1.0, laplacian_bgr, 1.0, 0)

    return sharpened


def denoise(img, kernel_size=3, strength=10):
    '''Applies Gaussian blur to reduce noise in an image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    - kernel_size (int): The size of the Gaussian kernel (default: 3).
    - strength (float): The strength of the blur (default: 10).
    
    Returns:
    - np.ndarray: The denoised image.

    Examples:
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> denoised_white_image = denoise(white_image)
    >>> np.array_equal(white_image, denoised_white_image)
    True

    >>> gradient_image = np.linspace(0, 255, 100).reshape(10, 10, 1).astype(np.uint8)
    >>> gradient_image = np.concatenate([gradient_image] * 3, axis=2)  # Make it 3-channel
    >>> denoised_gradient_image = denoise(gradient_image, kernel_size=5, strength=5)
    >>> np.array_equal(gradient_image, denoised_gradient_image)
    False
    '''
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), strength)
    return blurred



def selected_enhancements(img):
    '''Applies a series of selected image enhancements to an input image.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    
    Returns:
    - np.ndarray: The enhanced image.
    
    Since this is just a wrapper function for other enhancement functions, we don't need to test it.'''
    clahe_image = clahe(img)
    gamma = gamma_correction(clahe_image, gamma=0.5)
    laplacian = laplacian_sharpening(gamma)

    return laplacian


def visualize_10_images(df, enhance=False):
    '''Visualizes 10 randomly sampled images from the dataset with bounding boxes drawn.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the image data and bounding box coordinates.
    - enhance (bool): A flag indicating whether to apply image enhancements (default: False).
    
    Returns:
    - None
    
    Since this function is primarily for visualization, we do not need to test it.'''
    # Group the data by image
    grouped = df.groupby("img_path")
    sampled_groups = random.sample(list(grouped), 10) #randomly sample 10 images

    for img_path, group in sampled_groups:
        # Construct the full image path
        img_path = "data/tiles/" + img_path

        bounding_boxes = group[["xmin", "ymin", "xmax", "ymax"]].to_dict("records")
        annotated_image = draw_bounding_boxes(img_path, bounding_boxes)
        if enhance:
            annotated_image = selected_enhancements(annotated_image)

        if annotated_image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_image)
            plt.title(f"Annotated Image: {img_path}")
            plt.axis("off")
            plt.show()


def white_area_checker(img, threshold=80):
    '''Checks if an image has more than a specified percentage of white area.
    
    Parameters:
    - img (np.ndarray): The input image as a NumPy array.
    - threshold (int): The threshold percentage of white area (default: 80).
    
    Returns:
    - bool: True if the image has more than the threshold percentage of white area, False otherwise.

    Examples:
    >>> white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    >>> white_area_checker(white_image)
    True

    >>> black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> white_area_checker(black_image)
    False

    >>> mixed_image = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> mixed_image[:, :8] = 255  # 80% white, 20% black
    >>> white_area_checker(mixed_image)
    False

    >>> white_area_checker(mixed_image, threshold=70)
    True
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    white_area = (np.sum(binary == 255) / binary.size) * 100

    return white_area > threshold

import pandas as pd

def remove_white_area_images(df):
    '''Removes images with more than 80% white area from the dataset.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the image data.
    
    Returns:
    - pd.DataFrame: The updated DataFrame with white area images removed.

    Since this just applies the above function to the dataset, we don't need to test it.
    '''
    to_remove = []
    # Group the data by image
    grouped = df.groupby("img_path")

    # Iterate over each group (each image) and check for white area
    for img_path, _ in grouped:
        img_path_full = "data/tiles/" + img_path
        img = cv2.imread(img_path_full)

        if white_area_checker(img):
            print(
                f"Image at {img_path_full} has more than 80% white area. Removing from dataset."
            )
            to_remove.append(img_path)

    df = df[~df["img_path"].isin(to_remove)]

    return df


def apply_enhancements(df):
    '''Applies selected image enhancements to all images in the dataset.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the image data.
    
    Returns:
    - None
    
    Again, since this just applies the above function to the dataset, we don't need to test it.'''
    for img_path in df["img_path"].unique():
        img_path_full = "data/tiles/" + img_path
        img = cv2.imread(img_path_full)

        enhanced_img = selected_enhancements(img)

        # Save the enhanced image in a folder
        enhanced_img_path = "data/tiles/processed/" + img_path
        cv2.imwrite(enhanced_img_path, enhanced_img)


"""_______________________________________________________________"""

"""
Post-Processing Utilities

This section contains functions that assist in the post-processing of image data, 
particularly in the context of environmental and ecological analysis. These utilities 
are designed to work with probability maps, species predictions, and other outputs 
from machine learning models. Key functionalities include:

- **Visualization**: Functions for visualizing images, probability maps, 
and species maps, including the ability to add legends and save visualizations.
- **Map Modifications**: Tools to modify species maps based on probability thresholds, 
helping to refine predictions and improve accuracy.
- **Evaluation Metrics**: Calculation of metrics such as the F1 score to evaluate the 
performance of models on species prediction tasks.
- **Species Selection**: Functions to select the dominant species within bounding boxes, 
with optional normalization based on species proportions.
- **Integration with DeepForest**: Functions to create binary images from DeepForest results 
and to integrate these results with species predictions for more comprehensive analysis.
- **AGB and Carbon Calculations**: Tools to calculate above-ground biomass (AGB) 
and carbon storage based on species predictions, aiding in ecological assessments.

These utilities are essential for refining model outputs, evaluating performance, and generating 
insightful visualizations in the context of species classification and biomass estimation tasks.
"""

def visualize_step(image, title, save_path=None, cmap="gray"):
    '''Visualizes an image with a specified title and colormap.
    
    Parameters:
    - image (np.ndarray): The input image as a NumPy array.
    - title (str): The title for the image.
    - save_path (str): The file path to save the image (default: None).
    - cmap (str): The colormap to use for visualization (default: 'gray').

    Returns:
    - None
    
    Since this is primarily a visualization function, we don't need to test it.'''
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
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.show()

    except Exception as e:
        print(f"Failed to visualize image. Error: {e}")


def show_image_with_legend(
    image, title, save_path=None, cmap=None, norm=None, legend_info=None
):
    '''Displays an image with a legend containing color information.

    Parameters:
    - image (np.ndarray): The input image as a NumPy array.
    - title (str): The title for the image.
    - save_path (str): The file path to save the image (default: None).
    - cmap (str): The colormap to use for visualization (default: None).
    - norm (matplotlib.colors.Normalize): The normalization for the colormap (default: None).
    - legend_info (list of dict): A list of dictionaries containing color and label information for the legend.

    Returns:
    - None
    
    Since this is primarily a visualization function, we don't need to test it.'''
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

        _, ax = plt.subplots(figsize=(8, 6))
        if title:
            ax.set_title(title, fontsize=16)

        _ = ax.imshow(image, cmap=cmap, norm=norm)
        ax.axis("off")

        plt.subplots_adjust(right=0.8)

        if legend_info:
            legend_patches = [
                patches.Patch(color=info["color"], label=info["label"])
                for info in legend_info
            ]
            legend = ax.legend(
                handles=legend_patches,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize="small",
            )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.show()

    except Exception as e:
        print(f"Failed to visualize image with legend. Error: {e}")


def set_visuals():
    '''
    Sets the colormap, normalization, and legend information for visualizations.
    
    Returns:
    - cmap (matplotlib.colors.ListedColormap): The colormap for visualization.
    - norm (matplotlib.colors.BoundaryNorm): The normalization for the colormap.
    - legend_info (list of dict): A list of dictionaries containing color and label information for the legend.

    Examples:
    >>> cmap, norm, legend_info = set_visuals()
    >>> isinstance(cmap, ListedColormap)
    True

    >>> legend_info[0] == {"color": "purple", "label": "Background"}
    True
    '''
    species_values = [0, 1, 2, 3, 4, 5]
    species_colors = ["purple", "red", "blue", "green", "yellow", "orange"]
    species_names = [
        "Background",
        "Musacea",
        "Guaba",
        "Cacao",
        "Mango",
        "Otra Variedad",
    ]
    cmap = ListedColormap(species_colors)
    norm = BoundaryNorm(species_values + [max(species_values) + 1], cmap.N)
    legend_info = [
        {"color": color, "label": name}
        for color, name in zip(species_colors, species_names)
    ]

    return cmap, norm, legend_info


def load_map(path):
    '''Loads a map from a file.
    
    Parameters:
    - path (str): The file path to the map.
    
    Returns:
    - np.ndarray: The loaded map.
    
    Since this is a simple file loading function, we don't need to test it.'''
    with bz2.BZ2File(path, "rb") as f:
        title = pickle.load(f)
    return title


def select_and_visualize_prob(prob_map, testing=False):
    '''
    Selects and visualizes the probability map with the highest overall probability.

    Parameters:
    - prob_map (np.ndarray): The probability map to visualize.
    - testing (bool): If True, skip visualization (default: False).

    Returns:
    - tuple: The overall maximum probability and its location.

    Examples:
    >>> prob_map = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.2, 0.3], [0.4, 0.9]]])
    >>> select_and_visualize_prob(prob_map, testing=True)
    Overall Maximum Probability: 90.00%
    Location of Overall Maximum Probability: (1, 1)
    (0.9, (1, 1))
    '''
    max_probabilities = np.max(prob_map.transpose(1, 2, 0), axis=2)

    # Find the overall maximum probability and location
    overall_max_probability = np.max(max_probabilities)
    max_location = np.unravel_index(
        np.argmax(max_probabilities, axis=None), max_probabilities.shape
    )

    overall_max_probability_percentage = overall_max_probability * 100

    print(f"Overall Maximum Probability: {overall_max_probability_percentage:.2f}%")
    print(f"Location of Overall Maximum Probability: {max_location}")

    if not testing:
        plt.figure(figsize=(10, 10))
        plt.imshow(max_probabilities, cmap="hot", interpolation="none")
        plt.colorbar(label="Max Probability")
        plt.scatter(
            max_location[1],
            max_location[0],
            color="blue",
            marker="o",
            s=100,
            label="Max Probability Location",
        )
        plt.legend()
        plt.title("Maximum Probability for Each Pixel with Overall Max Location")
        plt.show()

    return overall_max_probability, max_location


def find_low_prob_pixels(probability_map):
    '''Finds the pixels with probability less than 0.2 in a probability map.

    Parameters:
    - probability_map (np.ndarray): The input probability map.

    Returns:
    - tuple: A tuple containing the indices of the low probability pixels.

    Examples:
    >>> prob_map = np.array([[[0.1, 0.15], [0.1, 0.4]], [[0.1, 0.6], [0.7, 0.1]], [[0.15, 0.3], [0.4, 0.1]]])
    >>> find_low_prob_pixels(prob_map)
    (array([0]), array([0]))
    '''
    max_probabilities = np.max(probability_map.transpose(1, 2, 0), axis=2)
    low_prob_pixels = np.where(max_probabilities < 0.2)
    return low_prob_pixels


def print_low_prob(probability_maps):
    '''Prints the percentage of pixels with probability less than 0.2 in each probability map.
    
    Parameters:
    - probability_maps (list of np.ndarray): A list of probability maps.
    
    Returns:
    - None
    
    Since this is primarily a print function, we don't need to test it.'''
    for probability_map in probability_maps:
        low_prob_pixels = find_low_prob_pixels(probability_map)
        total_pixels = probability_map.shape[1] * probability_map.shape[2]
        low_prob_pixel_count = len(low_prob_pixels[0])
        percentage_low_prob_pixels = (low_prob_pixel_count / total_pixels) * 100
        print(
            f"Percentage of Pixels with Probability < 0.2: {percentage_low_prob_pixels:.2f}%"
        )


def modify_species_map(species_map, low_prob_pixels, null_value=-1):
    '''Modifies the species map by setting the species values at low probability pixels to a null value.
    
    Parameters:
    - species_map (np.ndarray): The input species map.
    - low_prob_pixels (tuple): A tuple containing the indices of the low probability pixels.
    - null_value (int): The value to set for low probability pixels (default: -1).
    
    Returns:
    - np.ndarray: The modified species map.
    
    This is also just a wrapper that applies the modification, so we don't need to test it.'''
    modified_species_map = np.copy(species_map)
    for channel in range(modified_species_map.shape[0]):
        modified_species_map[channel, low_prob_pixels[0], low_prob_pixels[1]] = (
            null_value
        )
    return modified_species_map


def apply_modification(probability_maps, species_maps):
    '''Applies modifications to the species maps based on the probability maps.
    
    Parameters:
    - probability_maps (list of np.ndarray): A list of probability maps.
    - species_maps (list of np.ndarray): A list of species maps.
    
    Returns:
    - list of np.ndarray: A list of modified species maps.
    
    Just like above, this function is a wrapper for the modification process, so we don't need to test it.'''
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

    Examples:
    >>> y_true = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]])
    >>> y_pred = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> calculate_f1_score(y_true, y_pred)
    0.7777777777777778
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    f1 = f1_score(y_true_flat, y_pred_flat, average="weighted")
    return f1


def calculate_average_f1(targets, species_maps_pred):
    '''Calculates the average F1 score for a list of true and predicted species maps.
    
    Parameters:
    - targets (list of np.ndarray): A list of true species maps.
    - species_maps_pred (list of np.ndarray): A list of predicted species maps.
    
    Returns:
    - float: The average F1 score.
    
    Examples:
    >>> targets = [np.array([[0, 1], [1, 1]]), np.array([[1, 0], [0, 0]])]
    >>> preds = [np.array([[0, 1], [1, 0]]), np.array([[1, 1], [0, 0]])]
    >>> calculate_average_f1(targets, preds)
    0.7666666666666667
    0.7666666666666667
    Average F1 Score: 0.7666666666666667
    0.7666666666666667
    '''
    average = 0
    for i in range(len(targets)):
        current = calculate_f1_score(targets[i], species_maps_pred[i])
        print(current)
        average += current
    average /= len(targets)
    print("Average F1 Score:", average)
    return average


def mark_bw_boxes(image_path, root_dir, bounding_boxes):
    '''Marks the bounding boxes on a white background image.

    Parameters:
    - image_path (str): The path to the image file.
    - root_dir (str): The root directory containing the image file.
    - bounding_boxes (list of list): A list of bounding boxes in the format [xmin, ymin, xmax, ymax].

    Returns:
    - np.ndarray or None: The annotated image with bounding boxes drawn on a white background.

    Examples:
    >>> root_dir = ''
    >>> image_path = image_path
    >>> bounding_boxes = [[10, 10, 50, 50], [60, 60, 100, 100]]
    >>> result = mark_bw_boxes(image_path, root_dir, bounding_boxes)
    >>> isinstance(result, np.ndarray)
    True
    '''
    full_image_path = os.path.join(root_dir, image_path)
    image = cv2.imread(full_image_path)
    if image is None:
        print(f"Image at {image_path} could not be loaded.")
        return None
    white_background = np.ones_like(image) * 255

    for bbox in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(
            white_background,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            (0, 0, 0),
            -1,
        )

    return white_background


def visualize_bw_boxes(image_path, df, root_dir, show=True, save_path=None):
    '''Visualizes bounding boxes on a white background image.
    
    Parameters:
    - image_path (str): The path to the image file.
    - df (pd.DataFrame): The DataFrame containing the bounding box coordinates.
    - root_dir (str): The root directory containing the image file.
    - show (bool): A flag to display the annotated image (default: True).
    - save_path (str): The file path to save the annotated image (default: None).
    
    Returns:
    - np.ndarray or None: The annotated image with bounding boxes drawn on a white background.
    
    Since this is primarily a visualization function, we don't need to test it.'''
    bounding_boxes = df[
        ["pred_box_xmin", "pred_box_ymin", "pred_box_xmax", "pred_box_ymax"]
    ].values.tolist()
    annotated_image = mark_bw_boxes(image_path, root_dir, bounding_boxes)

    if show:
        if annotated_image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_image)
            plt.axis("off")
            plt.title("Annotated Image with Black Boxes on White Background")
            plt.show()

            if save_path:
                plt.close()

                annotated_image = Image.fromarray(annotated_image)
                annotated_image.save(save_path)
        else:
            print("Failed to annotate image.")

    return annotated_image


def create_binary_deepforest(deepforest_results_grouped, root_dir, show=True):
    '''Creates binary images from DeepForest results.

    Parameters:
    - deepforest_results_grouped (list of tuple): A list of tuples containing the image path and group of bounding boxes.
    - root_dir (str): The root directory containing the image files.
    - show (bool): A flag to display the binary images (default: True).

    Returns:
    - list of np.ndarray: A list of binary images.
    
    This is really just a function to apply bw box drawing to a list of images, so we don't need to test it.'''
    binary_image_list = []
    for image_path, group in deepforest_results_grouped:
        binary_box = visualize_bw_boxes(image_path, group, root_dir, show)
        bw_binary = cv2.cvtColor(binary_box, cv2.COLOR_BGR2GRAY)
        binary_image_list.append(bw_binary)

    return binary_image_list

'''Species proportions for the whole dataset to be used in normalization.'''
species_proportions = {
    1: 0.3241,  # Musacea
    2: 0.1287,  # Guaba
    3: 0.4354,  # Cacao
    5: 0.0922,  # Otra variedad
    4: 0.0192,  # Mango
}

def select_dominant_species(
    colored_image,
    df_boxes,
    species_proportions=None,
    bias_factor=1,
    normalize=True,
    return_type="image",
):
    '''Selects the dominant species within bounding boxes based on pixel counts.
    
    Parameters:
    - colored_image (np.ndarray): The colored image containing species labels.
    - df_boxes (pd.DataFrame): The DataFrame containing the bounding box coordinates.
    - species_proportions (dict): A dictionary containing the proportions of each species (default: None).
    - bias_factor (float): The bias factor for inverse proportion calculation (default: 1).
    - normalize (bool): A flag to normalize species counts based on proportions (default: True).
    
    Returns:
    - np.ndarray or pd.DataFrame: The result image or DataFrame with predicted species.
    
    Examples:
    >>> colored_image = np.array([[1, 1, 2], [2, 2, 3], [3, 3, 4]])
    >>> df_boxes = pd.DataFrame({
    ...     'pred_box_xmin': [0, 1],
    ...     'pred_box_ymin': [0, 1],
    ...     'pred_box_xmax': [2, 2],
    ...     'pred_box_ymax': [2, 2]
    ... })
    >>> species_proportions = {1: 0.7, 2: 0.2, 3: 0.1}
    >>> select_dominant_species(colored_image, df_boxes, species_proportions)
    array([[2, 2, 0],
           [2, 2, 0],
           [0, 0, 0]])
    '''
    labels = np.unique(colored_image)
    
    # Calculate inverse proportions only if species_proportions is provided
    if species_proportions is not None:
        inverse_proportions = {label: (1.0 / prop) ** bias_factor for label, prop in species_proportions.items() if prop > 0}
        min_inverse = min(inverse_proportions.values(), default=1)
        inverse_proportions = {label: inverse_proportions.get(label, min_inverse / 10) for label in labels}
    else:
        inverse_proportions = {}
    
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

        if normalize:
            normalized_counts = [count * inverse_proportions.get(species, 1) for species, count in zip(species_in_box, counts)]
        else:
            normalized_counts = counts
        
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


def process_images(deep_forest_df, species_map_pred):
    '''
    Combine images using DeepForest results and species predictions.
    
    Parameters:
    - deep_forest_df (pd.DataFrame): The DataFrame containing DeepForest results.
    - species_map_pred (list of np.ndarray): A list of predicted species maps.
    
    Returns:
    - tuple: A tuple containing the results and normalized results.
    
    Examples:
    >>> deep_forest_df = pd.DataFrame({
    ...     'pred_box_xmin': [0, 1],
    ...     'pred_box_ymin': [0, 1],
    ...     'pred_box_xmax': [2, 2],
    ...     'pred_box_ymax': [2, 2]
    ... }).groupby(np.array([0, 0]))
    >>> species_map_pred = [np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])]
    >>> species_proportions = {1: 0.7, 2: 0.2, 3: 0.1}
    >>> results, results_normalized = process_images(deep_forest_df, species_map_pred)
    >>> np.array_equal(results[0], np.array([[1, 1, 0], [1, 2, 0], [0, 0, 0]]))
    True
    >>> np.array_equal(results_normalized[0], np.array([[2, 2, 0], [2, 2, 0], [0, 0, 0]]))
    True
    '''
    results = []
    results_normalized = []

    for i in range(len(species_map_pred)):
        group_keys = list(deep_forest_df.groups.keys())
        curr_key = group_keys[i]
        df = deep_forest_df.get_group(curr_key)
        species_tester = species_map_pred[i]

        result_image = select_dominant_species(species_tester, df)

        result_image_normalized = select_dominant_species(
            species_tester, df, species_proportions
        )

        results.append(result_image)
        results_normalized.append(result_image_normalized)

    return results, results_normalized


def species_and_deepforest_df(species_maps_pred, grouped_merged):
    '''Combine species predictions and DeepForest and output results into a DataFrame.

    Parameters:
    - species_maps_pred (list of np.ndarray): A list of predicted species maps.
    - grouped_merged (list of tuple): A list of tuples containing the image path and group of bounding boxes.

    Returns:
    - pd.DataFrame: The DataFrame containing the combined results.
    '''
    all_results = []
    #see groups 
    keys = grouped_merged.groups.keys()

    for i, key in enumerate(keys):
        df = grouped_merged.get_group(key)
        species_tester = species_maps_pred[i]
        result_df = select_dominant_species(species_tester, df, return_type='dataframe')
        all_results.append(result_df)

    final_df = pd.concat(all_results, ignore_index=True)

    # Define a mapping for species
    species_dict = {
        1: "Musacea",
        2: "Guaba",
        3: "Cacao",
        4: "Mango",
        5: "Otra Variedad",
    }

    # Map the predicted species using the species_dict
    final_df["predicted_species"] = final_df["predicted_species"].map(species_dict)

    return final_df

def change_column_names(df):
    '''Change column names for use with different model outputs. 
    
    Parameters: 
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with updated column names

    Since this is a really simple column renaming function, we don't need to test it.
    '''
    df.rename(
        columns={
            "xmin": "true_box_xmin",
            "ymin": "true_box_ymin",
            "xmax": "true_box_xmax",
            "ymax": "true_box_ymax",
            "img_path": "image_path",
        },
        inplace=True,
    )
    return df

def change_column_names_back(df):
    '''Change column names back to original names.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with updated column names.
    
    Since this is a really simple column renaming function, we don't need to test it.'''
    df.rename(
        columns={
            "true_box_xmin": "xmin",
            "true_box_ymin": "ymin",
            "true_box_xmax": "xmax",
            "true_box_ymax": "ymax",
            "image_path": "img_path",
        },
        inplace=True,
    )
    return df


def calculate_AGB_averages(df, species_df):
    '''Prints the average AGB values for the dataset and species-matched dataset.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the AGB data.
    - species_df (pd.DataFrame): The DataFrame containing the species-matched AGB data.
    
    Returns:
    - None
    
    Since this is primarily a print function, we don't need to test it.'''

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
    '''Prints the average AGB values by image for the dataset and species-matched dataset.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the AGB data.
    - species_df (pd.DataFrame): The DataFrame containing the species-matched AGB data.
    
    Returns:
    - None
    
    Since this is primarily a print function, we don't need to test it.'''
    print("Including all species")
    print("=====================================")
    print(df.groupby("img_path")["AGB_difference"].mean())

    print(" ")

    print("Excluding species that don't match their predicted species")
    print("=====================================")
    print(species_df.groupby("img_path")["AGB_difference"].mean())


def sum_tables(df, species_df):
    '''Prints the sum of AGB and predicted AGB values for the dataset and species-matched dataset.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the AGB data.
    - species_df (pd.DataFrame): The DataFrame containing the species-matched AGB data.
    
    Returns:
    - None
    
    Since this is primarily a print function, we don't need to test it.'''
    AGB_sum = df.groupby("img_path")["AGB"].sum()
    pred_sum = df.groupby("img_path")["predicted_AGB"].sum()

    # display side by side
    all = pd.concat([AGB_sum, pred_sum], axis=1)
    print("Including all species")
    print(all)

    print(" ")

    AGB_sum = species_df.groupby("img_path")["AGB"].sum()
    pred_sum = species_df.groupby("img_path")["predicted_AGB"].sum()

    # display side by side
    species = pd.concat([AGB_sum, pred_sum], axis=1)
    print("Excluding species that don't match their predicted species")
    print(species)


def carbon_tables(df, species_df):
    '''Prints the sum of carbon values for the dataset and species-matched dataset.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the carbon data.
    - species_df (pd.DataFrame): The DataFrame containing the species-matched carbon data.
    
    Returns:
    - None
    
    Since this is primarily a print function, we don't need to test it.'''
    total_biomass = df.groupby("img_path")["AGB"].sum() * 1.22
    total_biomass_pred = df.groupby("img_path")["predicted_AGB"].sum() * 1.22

    carbon_AGB = total_biomass * 0.5
    carbon_pred = total_biomass_pred * 0.5
    absolute_difference = carbon_pred - carbon_AGB
    relative_difference = (absolute_difference / carbon_AGB).abs()

    # Combine all results into a DataFrame
    all_data = pd.concat([carbon_AGB, carbon_pred, absolute_difference, relative_difference], axis=1)
    all_data.columns = ['Carbon AGB', 'Carbon Predicted', 'Absolute Difference', 'Relative Difference']

    # Calculation for the species-matched dataset
    total_biomass_species = species_df.groupby("img_path")["AGB"].sum() * 1.22
    total_biomass_pred_species = species_df.groupby("img_path")["predicted_AGB"].sum() * 1.22

    carbon_AGB_species = total_biomass_species * 0.5
    carbon_pred_species = total_biomass_pred_species * 0.5
    absolute_difference_species = carbon_pred_species - carbon_AGB_species
    relative_difference_species = (absolute_difference_species / carbon_AGB_species).abs()

    # Combine all results into a DataFrame
    species_data = pd.concat([carbon_AGB_species, carbon_pred_species, absolute_difference_species, relative_difference_species], axis=1)
    species_data.columns = ['Carbon AGB', 'Carbon Predicted', 'Absolute Difference', 'Relative Difference']

    # Combine both DataFrames into a single one with multi-index for clarity
    combined_data = pd.concat([all_data, species_data], keys=['All Species', 'Species Matched'])

    return combined_data


"""_______________________________________________________________"""

"""
Allometric Equations Utilities

This module provides functions for calculating the Above-Ground Biomass (AGB) 
for different tree species based on their diameter. These calculations are 
based on species-specific allometric models, which are essential for ecological 
and environmental analyses, particularly in estimating biomass and carbon 
storage.

Key functionalities include:

- Species-Specific Models: Each tree species has its own allometric equation 
  to calculate AGB. These models are derived from empirical data and are 
  specific to the species' growth patterns and characteristics.

- Model Loading: Utilities to load saved GAM models from disk, ensuring that 
  predictions can be made using pre-trained models without the need to retrain 
  or redefine the models.
"""

def load_gam_model(species, model_directory):
    '''Helper function to load a GAM model from the specified directory.
    
    Parameters:
    - species (str): The species name.
    - model_directory (str): The directory containing the GAM models.
    
    Returns:
    - GAM model or None: The loaded GAM model.'''
    model_path = os.path.join(model_directory, f"{species}_gam_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        return None


def calculate_AGB(diameter, species, model_directory="models/allometric_models/"):
    '''
    Calculate the above-ground biomass (AGB) for a given diameter and species.

    Parameters:
    - diameter (float): The diameter of the tree.
    - species (str): The species name.
    - model_directory (str): The directory containing the GAM models (default: 'models/allometric_models/').

    Returns:
    - float: The predicted AGB.

    Examples:
    >>> diameter = 10.345047
    >>> species = "guaba"
    >>> result = calculate_AGB(diameter, species)
    >>> abs(result - 37.067309) < 1e-3
    True

    >>> diameter = 17.761650
    >>> species = "musacea"
    >>> result = calculate_AGB(diameter, species)
    >>> abs(result - 13.754021) < 1e-3
    True
    '''
    # Regularize the species name
    species = species.lower()

    # Try to load the GAM model for the species
    model = load_gam_model(species, model_directory)

    return model.predict(np.array([[diameter]]))[0]

'''_______________________________________________________________'''

#Giving the module a main method to run the doctests

if __name__ == "__main__":
    import doctest
    doctest.testmod()

