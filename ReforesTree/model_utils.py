# model_utils.py
"""
Model Utilities for Tree Crown Analysis

This module contains various utility functions and classes that support the 
training, evaluation, and visualization of models related to tree crown 
analysis. It covers a range of tasks, from calculating Intersection over Union 
(IoU) for bounding boxes to fitting different types of regression models for 
above-ground biomass (AGB) prediction.

The utilities are organized into several sections:

1. **Helper Functions for DeepForest**: Functions for calculating IoU, drawing 
   bounding boxes, and generating IoU results for tree crown analysis.

2. **Diameter Model Utilities**: Functions for preparing data, removing outliers, 
   and fitting models that predict DBH from various input features.

3. **SEDD Model Utilities**: Classes and functions specifically designed for the 
   SEDD model, which combines deep learning-based semantic segmentation with 
   distance map regression for tree crown detection and analysis.

4. **Allometric Relationship Visualization**: Tools for visualizing the relationship 
   between DBH and AGB, including the fitting and evaluation of various 
   regression models (linear, log-log, exponential, polynomial, GAM, etc.).

5. **Cross-Validation Utilities**: Functions to perform cross-validation for different 
   models, allowing the evaluation of model performance on unseen data.

This module is integral to the development and evaluation of models aimed at 
analyzing and understanding tree crown data, particularly for calculating carbon
sequestration potential.
"""

#this is a test image path to be used in the doctest
image_path = 'data/tiles/processed/Carlos Vera Arteaga RGB_0_0_0_4000_4000.png'

# Standard Library Imports
import os
import random
import bz2
import pickle
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

# Third-Party Library Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import joblib
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import torch.nn.functional as Func
from torchvision.transforms import functional as F
from pygam import LinearGAM, s


"""_______________________________________________________________________________________________________________________"""
"""
DeepForest Training Utilities

This section provides utility functions specifically tailored for training and 
evaluating models using the DeepForest framework. These utilities are designed 
to handle tasks such as calculating the Intersection over Union (IoU) for bounding 
boxes, generating IoU results, and visualizing the comparison between true and 
predicted bounding boxes.

The functions included in this section facilitate the following:

- **IoU Calculation**: Compute the Intersection over Union between two bounding boxes, 
  a critical metric for evaluating the accuracy of object detection models.

- **IoU Results Generation**: Aggregate IoU results across a dataset, allowing 
  for a comprehensive evaluation of model performance in predicting tree crown 
  bounding boxes.

- **Bounding Box Visualization**: Draw and compare true and predicted bounding boxes 
  on images to visually assess the accuracy of the model's predictions.

These utilities are essential for the effective training, evaluation, and refinement 
of DeepForest models in the context of tree crown analysis.
"""

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: A list or tuple with the format [xmin, ymin, xmax, ymax].
        box2: A list or tuple with the format [xmin, ymin, xmax, ymax].

    Returns:
        IoU: The Intersection over Union value.

    Examples:
        >>> calculate_iou([0, 0, 2, 2], [1, 1, 3, 3])
        0.14285714285714285
        >>> calculate_iou([0, 0, 2, 2], [2, 2, 3, 3])
        0.0
        >>> calculate_iou([0, 0, 3, 3], [1, 1, 2, 2])
        0.1111111111111111
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculate the area of intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No intersection

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    return iou


def generate_iou_results(test_data_grouped, image_path_to_index, list_of_predictions):
    '''Generate IoU results for the test dataset.
    
    Parameters:
    - test_data_grouped (DataFrameGroupBy): Grouped test data by image path.
    - image_path_to_index (dict): Dictionary mapping image paths to their corresponding indices.
    - list_of_predictions (list): List of predicted bounding boxes for each image.
    
    Returns:
    - iou_results (list): List of dictionaries containing IoU results for each true bounding box.
    
    This would be challenging to test without a full dataset, and individual function is tested just above.'''
    iou_results = []
    for group_key, group in test_data_grouped:
        image_path = group_key

        # Find the index for the current image path using the reversed dictionary
        number_of_image = image_path_to_index.get(image_path)

        if number_of_image is None:
            print(f"Image path {image_path} not found in the predictions list.")
            continue

        # Access the corresponding predictions for the current image
        curr = list_of_predictions[number_of_image]

        # Iterate over each true bounding box in the ground truth
        for _, true_box in group.iterrows():
            true_bbox = [
                true_box["xmin"],
                true_box["ymin"],
                true_box["xmax"],
                true_box["ymax"],
            ]
            highest_iou = 0.0
            best_prediction = None

            for _, pred_box in curr.iterrows():
                pred_bbox = [
                    pred_box["xmin"],
                    pred_box["ymin"],
                    pred_box["xmax"],
                    pred_box["ymax"],
                ]
                iou = calculate_iou(true_bbox, pred_bbox)

                # Check if this IoU is the highest seen so far
                if iou > highest_iou:
                    highest_iou = iou
                    best_prediction = pred_box

            if best_prediction is not None:
                iou_results.append(
                    {
                        "true_box_xmin": true_bbox[0],
                        "true_box_ymin": true_bbox[1],
                        "true_box_xmax": true_bbox[2],
                        "true_box_ymax": true_bbox[3],
                        "pred_box_xmin": best_prediction["xmin"],
                        "pred_box_ymin": best_prediction["ymin"],
                        "pred_box_xmax": best_prediction["xmax"],
                        "pred_box_ymax": best_prediction["ymax"],
                        "image_path": image_path,
                        "iou": highest_iou,
                    }
                )
            else:
                # Print details when no best prediction is found
                print(
                    f"No best prediction found for True Box: {true_bbox} in Image: {image_path}"
                )
    return pd.DataFrame(iou_results)


def draw_comparison_boxes(image_path, root_dir, true_boxes, pred_boxes):
    """
    Draws true and predicted bounding boxes on an image.

    Parameters:
        image_path (str): The path to the image file.
        true_boxes (list of list): List of true bounding boxes [xmin, ymin, xmax, ymax].
        pred_boxes (list of list): List of predicted bounding boxes [xmin, ymin, xmax, ymax].
        save_path (str, optional): If provided, the annotated image will be saved to this path.

    Returns:
        None: The annotated image is displayed using matplotlib.

    Since this is primarily a visualization function, we don't need to test it using docstrings.
    """
    image_path_full = os.path.join(root_dir, image_path)
    image = cv2.imread(image_path_full)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Draw true bounding boxes in green
    for box in true_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(
            image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 20
        )  

    # Draw predicted bounding boxes in blue
    for box in pred_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(
            image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 20
        )  

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Annotated Image")
    plt.show()


"""_______________________________________________________________________________________________________________________"""
"""
Diameter Model Utilities

This section provides utility functions for building, training, and evaluating models 
that predict DBH. These utilities facilitate the preparation of data, 
feature engineering, model training, and performance evaluation.

The functions included in this section support the following tasks:

- **Bounding Box Feature Engineering**: Calculate additional features such as bounding box 
  area, diagonal, and across dimensions, which are crucial for improving model predictions.

- **Data Preparation and Splitting**: Clean and preprocess data, including outlier removal 
  and optional one-hot encoding, followed by splitting the data into training and testing 
  sets.

- **Model Training**: Train various models including baseline neural networks, 
  Support Vector Regression (SVR), and Convolutional Neural Networks (CNNs) for predicting 
  diameter at base height.

- **Model Evaluation**: Evaluate the performance of trained models using metrics like Mean 
  Squared Error (MSE) and Root Mean Squared Error (RMSE), as well as visualizing the 
  results with scatter plots and comparison tables.

These utilities are designed to streamline the process of building robust models for 
accurately predicting diameter at base height.
"""


def add_bbox_columns(df):
    '''Add bounding box-related columns to the dataframe.
    
    Parameters:
    - df (DataFrame): The input dataframe containing bounding box coordinates.
    
    Returns:
    - df (DataFrame): The dataframe with additional bounding box-related columns.

    Examples:
    >>> import pandas as pd
    >>> data = {'xmin': [1, 2, 3], 'ymin': [1, 2, 3], 'xmax': [4, 5, 6], 'ymax': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> df = add_bbox_columns(df)
    >>> df['bbox_area'].tolist()
    [9, 9, 9]
    >>> df['bbox_diagonal'].tolist()
    [4.242640687119285, 4.242640687119285, 4.242640687119285]
    >>> df['bbox_across'].tolist()
    [3, 3, 3]
    '''
    # throw an error if the necessary columns are not present
    try:
        if (
            "xmin" not in df.columns
            or "xmax" not in df.columns
            or "ymin" not in df.columns
            or "ymax" not in df.columns
        ):
            raise ValueError("The necessary columns are not present in the dataframe")
        df["bbox_area"] = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
        df["bbox_diagonal"] = (
            (df["xmax"] - df["xmin"]) ** 2 + (df["ymax"] - df["ymin"]) ** 2
        ) ** 0.5

        # bbox diameter as either the height or width, whichever is larger
        df["bbox_across"] = np.where(
            df["xmax"] - df["xmin"] > df["ymax"] - df["ymin"],
            df["xmax"] - df["xmin"],
            df["ymax"] - df["ymin"],
        )
    except ValueError as e:
        print(e)
    return df


def remove_outliers(df, column_name):
    '''Remove outliers; anything outside of 10% and 90% quantiles based on the column.
    
    Parameters:
    - df (DataFrame): The input dataframe.
    - column_name (str): The column name to use for outlier removal.
    
    Returns:
    - df (DataFrame): The dataframe with outliers removed based on the specified column.

    Examples:
    >>> import pandas as pd
    >>> data = {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    >>> df = pd.DataFrame(data)
    >>> df_filtered = remove_outliers(df, 'values')
    >>> df_filtered['values'].tolist()
    [2, 3, 4, 5, 6, 7, 8, 9]
    '''
    df = df[
        (df[column_name] > df[column_name].quantile(0.10))
        & (df[column_name] < df[column_name].quantile(0.90))
    ]
    return df


def prepare_data(df, column_name, one_hot_encode=True):
    '''Prepare the data for training a diameter prediction model.
    
    Parameters:
    - df (DataFrame): The input dataframe containing the necessary columns.
    - column_name (str): The column name to use for outlier removal.
    - one_hot_encode (bool): Whether to perform one-hot encoding on the 'name' column.
    
    Returns:
    - df (DataFrame): The preprocessed dataframe ready for model training.
    
    Since this is primarily a wrapper function, using functions tested above and a pandas function, it doesn't need testing.'''
    df = add_bbox_columns(df)
    if one_hot_encode is True:
        df = pd.get_dummies(df, columns=["name"], dtype=int)
    df = remove_outliers(df, column_name)
    return df


def split_data(df, scaler):
    '''Split the data into features and target variables, and scale the features.
    
    Parameters:
    - df (DataFrame): The input dataframe containing the necessary columns.
    - scaler (bool or StandardScaler): Whether to use a StandardScaler for feature scaling, or provide a pre-fitted scaler.
    
    Returns:
    - X_process_scaled (ndarray): The scaled feature matrix.
    - y_process (ndarray): The target variable array.

    Examples:
    >>> import pandas as pd
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = {
    ...     "bbox_area": [100, 200, 300],
    ...     "bbox_diagonal": [10, 20, 30],
    ...     "bbox_across": [5, 10, 15],
    ...     "name_Cacao": [1, 0, 0],
    ...     "name_Musacea": [0, 1, 0],
    ...     "name_Guaba": [0, 0, 1],
    ...     "name_Mango": [0, 0, 0],
    ...     "name_Otra variedad": [0, 0, 0],
    ...     "diameter": [5.0, 10.0, 15.0]
    ... }
    >>> df = pd.DataFrame(data)
    >>> scaler = StandardScaler().fit(df.drop(columns=['diameter']))
    >>> X_scaled, y = split_data(df, scaler)
    >>> X_scaled.round(2)
    array([[-1.22, -1.22, -1.22,  1.41, -0.71, -0.71,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  , -0.71,  1.41, -0.71,  0.  ,  0.  ],
           [ 1.22,  1.22,  1.22, -0.71, -0.71,  1.41,  0.  ,  0.  ]])
    >>> y
    array([ 5., 10., 15.])
    '''
    # load the scaler
    if scaler is True:
        scaler = joblib.load("pkl_files/diameter_scaler.pkl")

    selected_columns = [
        "bbox_area",
        "bbox_diagonal",
        "bbox_across",
        "name_Cacao",
        "name_Musacea",
        "name_Guaba",
        "name_Mango",
        "name_Otra variedad",
        "diameter",
    ]
    if not all(col in df.columns for col in selected_columns):
        raise ValueError("The necessary columns are not present in the dataframe")

    X_process = df[
        [
            "bbox_area",
            "bbox_diagonal",
            "bbox_across",
            "name_Cacao",
            "name_Musacea",
            "name_Guaba",
            "name_Mango",
            "name_Otra variedad",
        ]
    ].values
    y_process = df["diameter"].values

    if scaler is False:
        scaler = StandardScaler()
        scaler.fit(X_process)
        joblib.dump(scaler, "pkl_files/diameter_scaler.pkl")

    X_process_scaled = scaler.transform(X_process)

    return X_process_scaled, y_process


def baseline_model(X_train):
    '''Create a baseline neural network model for diameter prediction.
    
    Parameters:
    - X_train (ndarray): The feature matrix for training the model.
    
    Returns:
    - model (Sequential): The compiled neural network model.
    
    This is a model so testing wouldn't really be applicable.'''
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer=Adam(), loss="mean_squared_error")

    return model


def mse_and_rmse(y_test, y_pred):
    '''Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
    
    Parameters:
    - y_test (ndarray): The true target variable values.
    - y_pred (ndarray): The predicted target variable values.
    
    Returns:
    - mse (float): The Mean Squared Error (MSE).
    - rmse (float): The Root Mean Squared Error (RMSE).

    Examples:
    >>> y_test = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> mse, rmse = mse_and_rmse(y_test, y_pred)
    Mean Squared Error: 0.375
    Root Mean Squared Error: 0.6123724356957945
    >>> round(mse, 3)
    0.375
    >>> round(rmse, 3)
    0.612
    '''
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")
    return mse, rmse


def plot_results(y_test, y_pred):
    '''Plot a scatter plot of the actual vs. predicted DBH.
    
    Parameters:
    - y_test (ndarray): The true target variable values.
    - y_pred (ndarray): The predicted target variable values.
    
    This is a visualization function, so testing isn't necessary.'''
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(y_test))
    plt.scatter(indices, y_test, color="blue", label="Actual Diameter", alpha=0.6)
    plt.scatter(indices, y_pred, color="red", label="Predicted Diameter", alpha=0.6)
    plt.xlabel("Index")
    plt.ylabel("Diameter")
    plt.title("Scatter Plot of Actual vs. Predicted DBH")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare(y_test, y_pred, head_or_tail="head"):
    '''Compare the actual and predicted DBH values.
    
    Parameters:
    - y_test (ndarray): The true target variable values.
    - y_pred (ndarray): The predicted target variable values.
    - head_or_tail (str): Whether to show the head or tail of the comparison.
    
    This is a visualization of results function, so testing isn't necessary.'''

    df_compare = pd.DataFrame({"Actual Diameter": y_test, "Predicted Diameter": y_pred})
    df_compare["Difference (Absolute)"] = abs(
        df_compare["Actual Diameter"] - df_compare["Predicted Diameter"]
    )
    if head_or_tail == "head":
        result = df_compare.head(20)
    if head_or_tail == "tail":
        result = df_compare.tail(20)
    print(result)


def SVR_model(X_train, y_train, X_val, y_val):
    '''Fit a Support Vector Regression (SVR) model for diameter prediction.
    
    Parameters:
    - X_train (ndarray): The feature matrix for training the model.
    - y_train (ndarray): The target variable array for training the model.
    - X_val (ndarray): The feature matrix for validation.
    - y_val (ndarray): The target variable array for validation.
    
    Returns:
    - model (SVR): The trained SVR model.
    
    This is a model so testing wouldn't really be applicable.'''
    # Define a range of hyperparameters for iteration
    C_values = [0.01, 0.1, 1, 10, 100]
    epsilon_values = [0.01, 0.1, 0.5, 1]
    gamma_values = ["scale", "auto", 0.1, 1]

    results = []

    for C in C_values:
        for epsilon in epsilon_values:
            for gamma in gamma_values:
                model = SVR(C=C, epsilon=epsilon, gamma=gamma)
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_val_pred)
                results.append((C, epsilon, gamma, mse))
                print(f"C={C}, epsilon={epsilon}, gamma={gamma}, MSE={mse}")

    # Find the best parameters
    best_params = sorted(results, key=lambda x: x[3])[0]
    print(
        f"Best parameters: C={best_params[0]}, epsilon={best_params[1]}, gamma={best_params[2]}, MSE={best_params[3]}"
    )

    return best_params


def best_SVR(best_params):
    '''Create the best SVR model based on the optimal hyperparameters.

    Parameters:
    - best_params (tuple): The optimal hyperparameters for the SVR model.

    Returns:
    - model (SVR): The SVR model with the best hyperparameters.
    
    This is a model so testing wouldn't really be applicable.'''
    model = SVR(C=best_params[0], epsilon=best_params[1], gamma=best_params[2])
    return model


def CNN(X_train):
    '''Create a Convolutional Neural Network (CNN) model for diameter prediction.
    
    Parameters:
    - X_train (ndarray): The feature matrix for training the model.
    
    Returns:
    - model (Sequential): The compiled CNN model.
    
    This is a model so testing wouldn't really be applicable.'''
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def xgboost(X_train, y_train, X_val, y_val):
    '''Fit an XGBoost model for diameter prediction.
    
    Parameters:
    - X_train (ndarray): The feature matrix for training the model.
    - y_train (ndarray): The target variable array for training the model.
    - X_val (ndarray): The feature matrix for validation.
    - y_val (ndarray): The target variable array for validation.
    
    Returns:
    - xg_model (XGBRegressor): The trained XGBoost model.
    
    This is a model so testing wouldn't really be applicable.'''
    xg_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=1,
        verbosity=1,
        random_state=42,
    )

    xg_model.fit(
        X_train,  # Use the scaled training data
        y_train,
        eval_set=[(X_val, y_val)],  # Including validation and optionally test set
        verbose=True,
    )

    return xg_model


"""_______________________________________________________________________________________________________________________"""

"""
SEDD Model Utilities

This section provides utility functions for the SEDD (Species and Distance Detection) 
model, which is designed for segmenting and analyzing tree crown images to predict 
species classification and distance maps. The SEDD model combines deep learning 
techniques with custom data preprocessing and augmentation strategies.

The functions included in this section support the following tasks:

- **Data Preparation and Augmentation**: Custom dataset classes and transformations 
  to handle tree crown images, including random cropping, flipping, rotation, and 
  color jittering to augment the training data and improve model generalization.

- **Model Architecture**: Definition of the SEDD model architecture, which includes 
  a ResNet18-based encoder and separate decoders for semantic segmentation and 
  distance map prediction. The model is designed to handle multi-task learning with 
  partially weighted focal loss for semantic segmentation and custom distance map 
  prediction.

- **Training and Validation**: Functions to train and validate the SEDD model, including 
  loss calculation, optimizer updates, and validation with a focus on efficient processing 
  of large image datasets through batch operations.

- **Inference and Evaluation**: Implementation of sliding window inference for handling 
  large images during prediction, as well as full evaluation of model performance, including 
  accuracy, precision, recall, F1 score, and mean squared error for distance maps.

- **Visualization**: Utilities for visualizing patches, species maps, and distance maps, 
  including custom color maps and legends for easy interpretation of model outputs.

These utilities are integral for training, evaluating, and deploying the SEDD model, 
which is critical for automated species classification and distance estimation in ITC.
"""

def draw_bounding_boxes(image_path, bounding_boxes, color=(255, 0, 0), thickness=20):
    '''Draw bounding boxes on an image.
    
    Parameters: 
    - image_path (str): The path to the image file.
    - bounding_boxes (list): A list of dictionaries containing bounding box coordinates.
    - color (tuple): The color of the bounding boxes.
    - thickness (int): The thickness of the bounding box lines.
    
    Returns:
    - image (ndarray): The image with the bounding boxes drawn.

    Examples:
    >>> bounding_boxes = [{"xmin": 100, "ymin": 150, "xmax": 200, "ymax": 250}]
    >>> image = draw_bounding_boxes(image_path, bounding_boxes)
    >>> isinstance(image, np.ndarray)
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
            image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness
        )

    return image


def split_data_SEDD(df, using_test=True):
    """
    Split the data into training, validation, and testing sets for the SEDD model.
    Data is in the form where each row corresponds to a specific tree, but there are
    multiple trees within a tile. This function ensures all trees in a given tile 
    are kept together.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the necessary columns.
    - using_test (bool): If True, split into training, validation, and test sets.
                         If False, only split into training and validation sets, and 
                         adjust sizes accordingly.

    Returns:
    - train_df (pd.DataFrame): The training set DataFrame.
    - val_df (pd.DataFrame): The validation set DataFrame.
    - test_df (pd.DataFrame, optional): The testing set DataFrame if using_test=True.

    Examples:
    >>> data = {
    ...     "img_path": ["tile1", "tile1", "tile2", "tile2", "tile3", "tile3", "tile4", "tile4", "tile5", "tile5"],
    ...     "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ... }
    >>> df = pd.DataFrame(data)
    >>> train_df, val_df, test_df = split_data_SEDD(df)
    Training set size: 6
    Validation set size: 2
    Testing set size: 2
    >>> len(train_df)
    6
    >>> len(val_df)
    2
    >>> len(test_df)
    2
    >>> train_df["img_path"].nunique()
    3
    >>> val_df["img_path"].nunique()
    1
    >>> test_df["img_path"].nunique()
    1

    >>> train_df, val_df = split_data_SEDD(df, using_test=False)
    Training set size: 6
    Validation set size: 4
    >>> len(train_df)
    6
    >>> len(val_df)
    4
    >>> train_df["img_path"].nunique()
    3
    >>> val_df["img_path"].nunique()
    2
    """
    
    tiles = df["img_path"].unique()
    np.random.seed(42)  # Ensures reproducibility
    np.random.shuffle(tiles)

    # Calculate original split indices
    train_end = int(0.7 * len(tiles))
    val_end = int(0.87 * len(tiles))

    if not using_test:        
        # Redistribute the test size proportionally between train and validation sets
        val_end = len(tiles)  # Remaining tiles go to validation

    # Split the tiles
    train_tiles = tiles[:train_end]
    val_tiles = tiles[train_end:val_end]
    
    train_df = df[df["img_path"].isin(train_tiles)]
    val_df = df[df["img_path"].isin(val_tiles)]
    
    if using_test:
        test_tiles = tiles[val_end:]
        test_df = df[df["img_path"].isin(test_tiles)]
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Testing set size: {len(test_df)}")
        return train_df, val_df, test_df
    else:
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        return train_df, val_df
    

def split_data_from_test_df(test_df, df):
    """
    Removes rows from `df` where the 'unique_id' is present in `test_df`.

    Parameters:
    test_df (pd.DataFrame): A DataFrame containing the 'unique_id' values to be removed.
    df (pd.DataFrame): The DataFrame from which rows will be removed.

    Returns:
    pd.DataFrame: A DataFrame with rows removed where 'unique_id' is in `test_df`.

    Example:
    >>> test_df = pd.DataFrame({'unique_id': [2, 3]})
    >>> df = pd.DataFrame({'unique_id': [1, 2, 3, 4], 'value': [10, 20, 30, 40]})
    >>> split_data_from_test_df(test_df, df)
       unique_id  value
    0          1     10
    3          4     40
    """
    test_ids = test_df['unique_id']
    # Remove rows with unique_id in test_ids
    df = df[~df['unique_id'].isin(test_ids)]
    return df

def generate_combined_maps(annotations, image_size, gaussian_sigma=2):
    '''Generate combined species and distance maps from annotations.
    
    Parameters:
    - annotations (DataFrame): The input dataframe containing tree annotations.
    - image_size (tuple): The size of the image.
    - gaussian_sigma (int): The standard deviation for the Gaussian filter.
    
    Returns:
    - species_map (ndarray): The species map.
    - distance_map (ndarray): The distance map.

    Examples:
    >>> data = {
    ...     "xmin": [10, 50],
    ...     "ymin": [10, 50],
    ...     "xmax": [20, 60],
    ...     "ymax": [20, 60],
    ...     "name": ["Musacea", "Guaba"]
    ... }
    >>> annotations = pd.DataFrame(data)
    >>> image_size = (100, 100)
    >>> species_map, distance_map = generate_combined_maps(annotations, image_size, gaussian_sigma=1)

    >>> species_map[15, 15]  # Should be 1 for 'Musacea'
    1
    >>> species_map[55, 55]  # Should be 2 for 'Guaba'
    2

    >>> distance_map[15, 15]  # Should be non-zero due to the distance from edge
    0.8557733
    >>> distance_map[55, 55]  # Should be non-zero due to the distance from edge
    0.8557733
    '''
    species_map = np.zeros(
        image_size, dtype=np.int64
    )  # Initialize with background (label 6)
    distance_map = np.zeros(image_size, dtype=np.float32)

    for _, row in annotations.iterrows():
        bbox = [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])]
        species_label = (
            1
            if row["name"] == "Musacea"
            else (
                2
                if row["name"] == "Guaba"
                else 3 if row["name"] == "Cacao" else 4 if row["name"] == "Mango" else 5
            )
        )  # 'Otra variedad' is 5
        species_map[bbox[1] : bbox[3], bbox[0] : bbox[2]] = species_label
        tree_mask = np.zeros(image_size, dtype=np.float32)
        tree_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1

        distances = distance_transform_edt(tree_mask)

        if np.max(distances) > 0:
            distances = distances / np.max(distances)

        distances = gaussian_filter(distances, sigma=gaussian_sigma)

        distance_map[bbox[1] : bbox[3], bbox[0] : bbox[2]] = distances[
            bbox[1] : bbox[3], bbox[0] : bbox[2]
        ]

    return species_map, distance_map


def how_much_tree(tensor):
    '''Calculate the proportion of tree pixels in a tensor.
    
    Parameters:
    - tensor (Tensor): The input tensor containing tree species labels.
    
    Returns:
    - proportion (float): The proportion of tree pixels in the tensor.

    Examples:
    >>> tensor = torch.tensor([[1, 0, 0],
    ...                        [0, 1, 0],
    ...                        [0, 0, 0]])
    >>> how_much_tree(tensor)
    0.2222222222222222

    >>> tensor = torch.tensor([[1, 1, 1],
    ...                        [1, 1, 1],
    ...                        [1, 1, 1]])
    >>> how_much_tree(tensor)
    1.0

    >>> tensor = torch.tensor([[0, 0, 0],
    ...                        [0, 0, 0],
    ...                        [0, 0, 0]])
    >>> how_much_tree(tensor)
    0.0
    '''
    pixel_num = torch.count_nonzero(tensor)
    total_elements = tensor.numel()
    return pixel_num.item() / total_elements


class RandomTransforms:
    '''Randomly apply transformations to the input image, species map, and distance map.'''
    def __init__(
        self,
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        rotation_degrees=(0, 90, 180, 270),
        color_jitter=None,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter

    def __call__(self, image, species_map, distance_map):
        '''Apply the random transformations to the input data. 
        
        Includes horizontal and vertical flips and rotation.

        Parameters:
        - image (Tensor): The input image tensor.
        - species_map (Tensor): The input species map tensor.
        - distance_map (Tensor): The input distance map tensor.

        Returns:
        - image (Tensor): The transformed image tensor.
        - species_map (Tensor): The transformed species map tensor.
        - distance_map (Tensor): The transformed distance map tensor.'''
        if random.random() < self.horizontal_flip_prob:
            image = F.hflip(image)
            species_map = F.hflip(species_map)
            distance_map = F.hflip(distance_map)

        if random.random() < self.vertical_flip_prob:
            image = F.vflip(image)
            species_map = F.vflip(species_map)
            distance_map = F.vflip(distance_map)

        if self.rotation_degrees:
            angle = random.choice(self.rotation_degrees)
            image = F.rotate(image, angle)
            species_map = F.rotate(species_map, angle, fill=0)
            distance_map = F.rotate(distance_map, angle, fill=0)

        return image, species_map, distance_map


class TreeCrownDataset(Dataset):
    '''Custom dataset class for tree crown images.'''
    def __init__(self, dataframe, root_dir, transform=None, num_crops=8):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.num_crops = num_crops
        self.image_groups = dataframe.groupby("img_path")

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        '''Get a batch of image crops, species maps, and distance maps applying crops on the fly.

        Parameters:
        - idx (int): The index of the batch.

        Returns:
        - crop_imgs (Tensor): The batch of image crops.
        - crop_species_maps (Tensor): The batch of species maps.
        - crop_dist_maps (Tensor): The batch of distance maps.'''

        image_path = list(self.image_groups.groups.keys())[idx]
        annotations = self.image_groups.get_group(image_path)

        img_full_path = os.path.join(self.root_dir, image_path)
        image = Image.open(img_full_path).convert("RGB")
        image = np.array(image)

        species_map, distance_map = generate_combined_maps(annotations, image.shape[:2])

        crop_size = 224
        height, width = image.shape[:2]

        crop_imgs = []
        crop_species_maps = []
        crop_dist_maps = []

        count = 0

        while count < self.num_crops:
            # Random cropping logic
            x = random.randint(0, width - crop_size)
            y = random.randint(0, height - crop_size)

            crop_img = image[y : y + crop_size, x : x + crop_size]
            crop_species = species_map[y : y + crop_size, x : x + crop_size]
            crop_dist = distance_map[y : y + crop_size, x : x + crop_size]

            crop_img = transforms.ToTensor()(crop_img)
            crop_species = torch.tensor(crop_species, dtype=torch.int64).unsqueeze(
                0
            )  # Add channel dimension
            crop_dist = torch.tensor(crop_dist, dtype=torch.float32).unsqueeze(
                0
            )  # Add channel dimension

            # Apply the same transform to species and distance maps if defined
            if self.transform:
                crop_img, crop_species, crop_dist = self.transform(
                    crop_img, crop_species, crop_dist
                )

            # Check if the crop contains at least 10% of the tree crown
            if how_much_tree(crop_species) > 0.1:
                crop_imgs.append(crop_img)
                crop_species_maps.append(crop_species)
                crop_dist_maps.append(crop_dist)
                count += 1

        crop_imgs = torch.stack(crop_imgs)
        crop_species_maps = torch.stack(crop_species_maps).squeeze(
            1
        )  # Remove extra channel dimension
        crop_dist_maps = torch.stack(crop_dist_maps).squeeze(1)

        return crop_imgs, crop_species_maps, crop_dist_maps


class TestDataset(Dataset):
    '''Custom dataset class for test images.'''
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.image_groups = dataframe.groupby("img_path")

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        '''Get a test image, species map, and distance map.

        Parameters:
        - idx (int): The index of the image.

        Returns:
        - image (Tensor): The test image.'''
        image_path = list(self.image_groups.groups.keys())[idx]
        annotations = self.image_groups.get_group(image_path)

        img_full_path = os.path.join(self.root_dir, image_path)
        image = Image.open(img_full_path).convert("RGB")

        # Generate combined maps
        species_map, distance_map = generate_combined_maps(
            annotations, image.size[::-1]
        )

        image = transforms.ToTensor()(image)
        species_map = torch.tensor(species_map, dtype=torch.int64).unsqueeze(
            0
        )  # Add channel dimension
        distance_map = torch.tensor(distance_map, dtype=torch.float32).unsqueeze(
            0
        )  # Add channel dimension

        # current transform is not used
        if self.transform:
            image, species_map, distance_map = self.transform(
                image, species_map, distance_map
            )

        species_map = species_map.squeeze(0)  # Remove extra channel dimension
        distance_map = distance_map.squeeze(0)

        return image, species_map, distance_map

'''visualization helper variables'''
cmap = plt.get_cmap("tab10", 6)
norm = mcolors.BoundaryNorm(
    np.arange(-0.5, 6.5, 1), cmap.N
)  


def patch(img_patch, species_patch, distance_patch, cmap="viridis"):
    """
    Visualizes a batch of image patches, species maps, and distance maps.

    Parameters:
        img_patch (torch.Tensor): A batch of image patches with shape [B, C, H, W].
        species_patch (torch.Tensor): A batch of species maps with shape [B, H, W].
        distance_patch (torch.Tensor): A batch of distance maps with shape [B, H, W].
        cmap (str): Colormap for the species map visualization.

    Returns:
        None

    This is a visualization function, so testing isn't necessary.
    """

    # Define normalization to ensure it covers all species (from 0 to 5)
    norm = Normalize(vmin=0, vmax=5)

    batch_size = img_patch.size(0)

    for i in range(batch_size):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # Permute the tensor from [C, H, W] to [H, W, C] for visualization
        single_img_patch = img_patch[i].permute(1, 2, 0).numpy()

        ax[0].imshow(single_img_patch)
        ax[0].set_title("Image Patch")
        ax[0].axis("off")

        sm = ax[1].imshow(
            species_patch[i], cmap=cmap, norm=norm
        )  # Apply the custom colormap and normalizer
        ax[1].set_title("Species Map")
        ax[1].axis("off")
        cbar = plt.colorbar(sm, ax=ax[1], ticks=np.arange(0, 6), spacing="proportional")
        cbar.set_ticklabels(
            [
                "Background",
                "Species 1",
                "Species 2",
                "Species 3",
                "Species 4",
                "Species 5",
            ]
        )  

        # Distance Map
        dm = ax[2].imshow(distance_patch[i], cmap="plasma")
        ax[2].set_title("Distance Map")
        ax[2].axis("off")
        plt.colorbar(dm, ax=ax[2])

        plt.tight_layout()
        plt.show()


def visualize_patches(data_iter, max_iterations=5):
    '''Visualize a batch of image patches, species maps, and distance maps.
    
    Parameters:
    - data_iter (DataLoader): The DataLoader object containing the dataset.
    - max_iterations: The maximum number of iterations to visualize.
    
    Returns:
    - None
    
    This is a visualization function, so testing isn't necessary.'''
    count = 0

    while count < max_iterations:
        try:
            # Get the next batch from the dataset
            img_patch, species_patch, distance_patch = next(data_iter)
            patch(img_patch[0:1], species_patch[0:1], distance_patch[0:1])

            count += 1
        except StopIteration:
            # Handle case where fewer items are available than expected
            print("Reached the end of the dataset")
            break


def show_images_with_legend(
    images,
    title,
    nrow=2,
    ncol=2,
    save_image=False,
    cmap=None,
    norm=None,
    legend_patches=None,
):
    '''Show a grid of images with a legend.
    
    Parameters:
    - images (list): The list of images to display.
    - title (str): The title of the plot.
    - nrow (int): The number of rows in the grid.
    - ncol (int): The number of columns in the grid.
    - save_image (bool): Whether to save the image.
    - cmap (str): The colormap for the images.
    - norm (Normalize): The normalization for the images.
    - legend_patches (list): The legend patches for the species map.
    
    Returns:
    - None
    
    This is a visualization function, so testing isn't necessary.'''
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
    if title:
        fig.suptitle(title, fontsize=16)

    for i, img in enumerate(images):
        if i >= nrow * ncol:
            break
        ax = axes[i // ncol, i % ncol]

        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.squeeze(img)

        # If img still has 3 dimensions and first dimension is batch size, select the first in the batch
        if img.ndim == 3 and img.shape[0] > 1:
            img = img[0]

        # Ensure img has 2 dimensions for cmap and norm
        ax.imshow(img, cmap=cmap, norm=norm)
        ax.axis("off")

    # Add the legend to the figure outside the subplots
    if legend_patches:
        fig.legend(handles=legend_patches, loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.show()

    if save_image:
        fig.savefig(title + ".png")


def show_images(images, title, save_image=False, nrow=2, ncol=2):
    '''Show a grid of images.

    Parameters:
    - images (list): The list of images to display.
    - title (str): The title of the plot.
    - save_image (bool): Whether to save the image.
    - nrow (int): The number of rows in the grid.
    - ncol (int): The number of columns in the grid.

    Returns:
    - None
    
    This is a visualization function, so testing isn't necessary.'''
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
            img = img.permute(
                1, 2, 0
            )  # Rearrange dimensions for matplotlib (C, H, W) -> (H, W, C)

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    if save_image:
        fig.savefig(title + ".png")


def set_visuals():
    '''Set the visual properties for the plots.
    
    Parameters:
    - None
    
    Returns:
    - cmap (ListedColormap): The colormap for the species map.
    - norm (BoundaryNorm): The normalization for the species map.
    - legend_patches (list): The legend patches for the species map.

    Examples:
    >>> cmap, norm, legend_patches = set_visuals()
    >>> cmap.colors
    ['purple', 'red', 'blue', 'green', 'yellow', 'orange']
    >>> norm.boundaries
    array([0, 1, 2, 3, 4, 5, 6])
    >>> len(legend_patches)
    6
    >>> legend_patches[0].get_label()
    'Background'
    >>> legend_patches[0].get_facecolor()
    (0.5019607843137255, 0.0, 0.5019607843137255, 1.0)
    >>> legend_patches[1].get_label()
    'Musacea'
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

    # Create the legend patches correctly
    legend_patches = [
        mpatches.Patch(color=color, label=name)
        for color, name in zip(species_colors, species_names)
    ]

    return cmap, norm, legend_patches


def analyze_class_distribution(dataloader):
    '''Analyze the class distribution of the dataset.
    
    Parameters:
    - dataloader (DataLoader): The DataLoader object containing the dataset.
    
    Returns:
    - class_distribution (dict): The distribution of each class in the dataset.

    It would be difficult to test this function as it requires a DataLoader object. Also, it is only used for analysis.
    '''
    all_targets = []

    with torch.no_grad():
        for _, targets, _ in dataloader:
            all_targets.append(targets.cpu().numpy())

    all_targets = np.concatenate(all_targets).ravel()
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
            print(
                "Cuda installed! Running on GPU {} {}!".format(
                    idx, torch.cuda.get_device_name(idx)
                )
            )
            device = "cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print(
                "Cuda installed but only {} GPU(s) available! Running on GPU 0 {}!".format(
                    torch.cuda.device_count(), torch.cuda.get_device_name()
                )
            )
            device = "cuda:0"
        else:
            device = "cpu"
            print("No GPU available! Running on CPU")
    return device


class ResNet18Encoder(nn.Module):
    '''ResNet18-based encoder for the SEDD model.'''
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-2]
        )  # Remove last layers

    def forward(self, x):
        return self.resnet(x)


class SemanticSegmentationDecoder(nn.Module):
    '''Semantic segmentation decoder for the SEDD model.'''
    def __init__(self, num_classes):
        super(SemanticSegmentationDecoder, self).__init__()
        self.deeplab_head = DeepLabHead(512, num_classes)
        self.dropout = nn.Dropout(p=0.65)  # Add dropout layer with a rate of 0.65

    def forward(self, x):
        x = self.deeplab_head(x)
        x = self.dropout(x)  # Apply dropout
        return x


class DistanceMapDecoder(nn.Module):
    '''Distance map decoder for the SEDD model.'''
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
    '''Partially weighted categorical focal loss for the SEDD model.'''
    def __init__(self, gamma=2, alpha=0.25, beta=0.75):
        super(PartiallyWeightedCategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        '''Compute the partially weighted categorical focal loss. Assumes background pixels are labeled as 0.
        
        Parameters:
        - input (Tensor): The input tensor.
        - target (Tensor): The target tensor.
        
        Returns:
        - loss (Tensor): The computed loss.'''
        mask = target != 0  
        ce_loss = nn.CrossEntropyLoss(reduction="none")(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        weights = torch.where(target == 0, self.alpha, self.beta)
        loss = weights * focal_loss
        masked_loss = loss * mask.float()  # Apply mask to focus on labeled pixels
        return masked_loss.sum() / mask.sum()
    
class PWCFLNonMask(nn.Module):
    '''Partially weighted categorical focal loss for the SEDD model without masking.'''
    def __init__(self, gamma=2, alpha=0.25, beta=0.75):
        super(PWCFLNonMask, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        '''Compute the partially weighted categorical focal loss without masking. 
        
        Parameters:
        - input (Tensor): The input tensor.
        - target (Tensor): The target tensor.

        Returns:
        - loss (Tensor): The computed loss.'''
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        weights = torch.where(target == 0, self.alpha, self.beta)
        loss = weights * focal_loss
        return loss.mean()


class SEDDModel(nn.Module):
    '''SEDD model for species and distance detection.'''
    def __init__(self, num_classes):
        super(SEDDModel, self).__init__()
        self.encoder = ResNet18Encoder()
        self.semantic_decoder = SemanticSegmentationDecoder(num_classes)
        self.distance_decoder = DistanceMapDecoder()

    def forward(self, x):
        '''Forward pass of the SEDD model.
        
        Parameters:
        - x (Tensor): The input tensor.
        
        Returns:
        - semantic_output (Tensor): The semantic segmentation output.'''
        encoded = self.encoder(x)
        semantic_output = self.semantic_decoder(encoded)
        distance_output = self.distance_decoder(encoded)

        semantic_output = torch.nn.functional.interpolate(
            semantic_output, size=(224, 224), mode="bilinear", align_corners=True
        )
        distance_output = torch.nn.functional.interpolate(
            distance_output, size=(224, 224), mode="bilinear", align_corners=True
        )

        return semantic_output, distance_output


def final_loss(semantic_loss, distance_loss):
    '''Compute the final loss for the SEDD model.
    
    Parameters:
    - semantic_loss (Tensor): The semantic loss.
    - distance_loss (Tensor): The distance loss.
    
    Returns:
    - loss (Tensor): The final loss.'''
    # weighing the semantic loss by 2
    return (2 * semantic_loss) + distance_loss

def final_loss_regularized(semantic_loss, distance_loss):
    '''Compute the final loss for the SEDD model with more regularization from the distance loss.

    Parameters:
    - semantic_loss (Tensor): The semantic loss.
    - distance_loss (Tensor): The distance loss.

    Returns:
    - loss (Tensor): The final loss.'''
    return semantic_loss + distance_loss

def fit(model, dataloader, optimizer, semantic_loss_fn, distance_loss_fn, device, regularized=False):
    '''Fit the SEDD model on the training data.
    
    Parameters:
    - model (SEDDModel): The SEDD model.
    - dataloader (DataLoader): The DataLoader object containing the training data.
    - optimizer (Optimizer): The optimizer for the model.
    - semantic_loss_fn (PartiallyWeightedCategoricalFocalLoss): The semantic loss function.
    - distance_loss_fn (MSELoss): The distance loss function.
    - device (str): The device to run the model on.
    - regularized (bool): Whether to use the regularized loss function.
    
    Returns:
    - train_loss (float): The training loss.'''
    model.train()
    running_loss = 0.0

    for images, targets, distance_maps in dataloader:

        # reshaping from 5D tensor (batches and crops) to 4D tensor (batches*crops)
        images = images.view(
            -1, images.size(2), images.size(3), images.size(4)
        )  # [B*ncrops, C, H, W]
        targets = targets.view(
            -1, targets.size(2), targets.size(3)
        )  # [B*ncrops, H, W]
        distance_maps = distance_maps.view(
            -1, distance_maps.size(2), distance_maps.size(3)
        )

        images = images.to(device)
        targets = targets.to(device)
        distance_maps = distance_maps.to(device)

        optimizer.zero_grad()
        semantic_outputs, distance_outputs = model(images)

        # Ensure target tensors are cast to appropriate types
        semantic_loss = semantic_loss_fn(semantic_outputs, targets.to(torch.long))
        distance_loss = distance_loss_fn(
            distance_outputs, distance_maps.to(torch.float32)
        )
        if regularized:
            loss = final_loss_regularized(semantic_loss, distance_loss)
        else: 
            loss = final_loss(semantic_loss, distance_loss)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader.dataset)
    return train_loss


def validate(model, dataloader, semantic_loss_fn, distance_loss_fn, device, regularized=False):
    '''Validate the SEDD model on the validation data.
    
    Parameters:
    - model (SEDDModel): The SEDD model.
    - dataloader (DataLoader): The DataLoader object containing the validation data.
    - semantic_loss_fn (PartiallyWeightedCategoricalFocalLoss): The semantic loss function.
    - distance_loss_fn (MSELoss): The distance loss function.
    - device (str): The device to run the model on.
    
    Returns:
    - val_loss (float): The validation loss.'''
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets, distance_maps in dataloader:
            # again reshaping from 5D tensor to 4D tensor
            images = images.view(
                -1, images.size(2), images.size(3), images.size(4)
            )  # [B*ncrops, C, H, W]
            targets = targets.view(
                -1, targets.size(2), targets.size(3)
            )  # [B*ncrops, H, W]
            distance_maps = distance_maps.view(
                -1, distance_maps.size(2), distance_maps.size(3)
            )

            images = images.to(device)
            targets = targets.to(device)
            distance_maps = distance_maps.to(device, dtype=torch.float32)

            semantic_outputs, distance_outputs = model(images)

            semantic_loss = semantic_loss_fn(semantic_outputs, targets.to(torch.long))
            distance_loss = distance_loss_fn(
                distance_outputs, distance_maps.to(torch.float32)
            )
            if regularized:
                loss = final_loss_regularized(semantic_loss, distance_loss)
            else: 
                loss = final_loss(semantic_loss, distance_loss)

            running_loss += loss.item()

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss


def sliding_window_inference(model, image, patch_size, stride, device, num_classes=6):
    '''Perform sliding window inference on a large image.
    
    Parameters:
    - model (SEDDModel): The SEDD model.
    - image (Tensor): The input image tensor.
    - patch_size (tuple): The size of the image patches.
    - stride (tuple): The stride for the sliding window.
    - device (str): The device to run the model on.
    - num_classes (int): The number of classes in the dataset.
    
    Returns:
    - semantic_prediction_map (Tensor): The semantic prediction map.
    - distance_prediction_map (Tensor): The distance prediction map.'''

    _, H, W = image.shape
    patches = []
    coords = []

    # Extract patches using sliding window
    for y in range(0, H - patch_size[0] + 1, stride[0]):
        for x in range(0, W - patch_size[1] + 1, stride[1]):
            patch = image[:, y : y + patch_size[0], x : x + patch_size[1]]
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
    for (y, x), semantic_pred, distance_pred in zip(
        coords, semantic_preds, distance_preds
    ):
        semantic_prediction_map[
            :, y : y + patch_size[0], x : x + patch_size[1]
        ] += semantic_pred
        distance_prediction_map[
            :, y : y + patch_size[0], x : x + patch_size[1]
        ] += distance_pred
        count_map[y : y + patch_size[0], x : x + patch_size[1]] += 1

    # Avoid division by zero
    count_map[count_map == 0] = 1  # To avoid division by zero
    count_map = count_map.unsqueeze(0).repeat(num_classes, 1, 1)
    semantic_prediction_map /= count_map
    distance_prediction_map /= count_map[0]

    return semantic_prediction_map, distance_prediction_map


def full_evaluation(model, dataloader, patch_size, stride, device):
    '''Perform full evaluation on the test data.

    Parameters:
    - model (SEDDModel): The SEDD model.
    - dataloader (DataLoader): The DataLoader object containing the test data.
    - patch_size (tuple): The size of the image patches.
    - stride (tuple): The stride for the sliding window.
    - device (str): The device to run the model on.

    Returns:
    - accuracy (float): The accuracy of the model.
    - precision (float): The precision of the model.
    - recall (float): The recall of the model.
    - f1 (float): The F1 score of the model.
    - mse (float): The mean squared error of the model.
    - all_original_images (list): The list of original images.
    - all_species_maps (list): The list of species maps.
    - all_distance_maps (list): The list of distance maps.
    - all_probability_maps (list): The list of probability maps.
    - all_distance_maps_true (list): The list of true distance maps.
    - all_species_maps_true (list): The list of true species
    '''
    model.eval()
    all_targets = []
    all_predictions = []
    all_distances_true = []
    all_distances_pred = []
    all_species_maps = []
    all_distance_maps = []
    all_probability_maps = []
    all_original_images = []
    all_distance_maps_true = []
    all_species_maps_true = []

    with torch.no_grad():
        for images, targets, distance_maps in dataloader:
            for i in range(images.shape[0]):
                image = images[i].to(device)
                target = targets[i].to(device)
                distance_map = distance_maps[i].to(device)

                # Perform sliding window inference
                semantic_pred, distance_pred = sliding_window_inference(
                    model, image, patch_size, stride, device
                )

                # Convert predictions to final class labels and append to lists
                predictions = torch.argmax(semantic_pred, dim=0)
                probabilities = Func.softmax(
                    semantic_pred, dim=0
                )  # Convert logits to probabilities

                all_targets.append(target.cpu().numpy().ravel())
                all_predictions.append(predictions.cpu().numpy().ravel())
                all_distances_true.append(distance_map.cpu().numpy().ravel())
                all_distances_pred.append(distance_pred.cpu().numpy().ravel())

                # Save images and maps
                all_species_maps_true.append(target.cpu().numpy())
                all_distance_maps_true.append(distance_map.cpu().numpy())
                all_original_images.append(images[i].cpu().numpy())
                all_species_maps.append(semantic_pred.cpu().numpy())
                all_distance_maps.append(distance_pred.cpu().numpy())
                all_probability_maps.append(probabilities.cpu().numpy())

                # Cleanup to reduce memory footprint
                del (
                    image,
                    target,
                    distance_map,
                    semantic_pred,
                    distance_pred,
                    predictions,
                    probabilities,
                )
                torch.cuda.empty_cache()

    # Compute overall metrics
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(all_targets, all_predictions, average="weighted", zero_division=0)
    mse = np.mean(
        (np.concatenate(all_distances_true) - np.concatenate(all_distances_pred)) ** 2
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        mse,
        all_original_images,
        all_species_maps,
        all_distance_maps,
        all_probability_maps,
        all_distance_maps_true,
        all_species_maps_true,
    )


def print_results_and_save(
    accuracy,
    precision,
    recall,
    f1,
    mse,
    all_original_images,
    all_species_maps,
    all_distance_maps,
    all_probability_maps,
    all_distance_maps_true,
    all_species_maps_true,
    overlap,
    starting_string = 'map_'
):
    '''Print the evaluation results and save the maps to disk.

    Parameters:
    - accuracy (float): The accuracy of the model.
    - precision (float): The precision of the model.
    - recall (float): The recall of the model.
    - f1 (float): The F1 score of the model.
    - mse (float): The mean squared error of the model.
    - all_original_images (list): The list of original images.
    - all_species_maps (list): The list of species maps.
    - all_distance_maps (list): The list of distance maps.
    - all_probability_maps (list): The list of probability maps.
    - all_distance_maps_true (list): The list of true distance maps.
    - all_species_maps_true (list): The list of true species maps.
    - overlap (float): The overlap percentage.

    Returns:
    - None'''
    print(f"Results with {overlap * 100}% overlap:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Distance Map MSE: {mse:.4f}")

    # Save the distance and species maps to disk
    with bz2.BZ2File(f"{starting_string}species_{overlap}.pkl", "w") as f:
        pickle.dump(all_species_maps, f)

    with bz2.BZ2File(f"{starting_string}distance_{overlap}.pkl", "w") as f:
        pickle.dump(all_distance_maps, f)

    with bz2.BZ2File(f"{starting_string}probability_{overlap}.pkl", "w") as f:
        pickle.dump(all_probability_maps, f)

    with bz2.BZ2File(f"{starting_string}original_{overlap}.pkl", "w") as f:
        pickle.dump(all_original_images, f)

    # Save the ground truth distance and probability maps
    with bz2.BZ2File(f"{starting_string}distance_true_{overlap}.pkl", "w") as f:
        pickle.dump(all_distance_maps_true, f)

    with bz2.BZ2File(f"{starting_string}species_true_{overlap}.pkl", "w") as f:
        pickle.dump(all_species_maps_true, f)


"""_______________________________________________________________________________________________________________________"""

"""
Allometric Equations Utilities

This section provides utility functions for calculating Above-Ground Biomass (AGB) 
using species-specific allometric equations. Allometric equations are essential in 
ecological and forestry studies for estimating the biomass and carbon storage of trees 
based on attributes like diameter.

The functions included in this section support the following tasks:

- **Model Fitting and Evaluation**: Multiple statistical models (log-log, linear, 
  exponential, logarithmic, polynomial, and Generalized Additive Models) are provided 
  for fitting species-specific allometric equations. These models are trained using 
  DBH as the predictor variable and AGB as the target.

- **Cross-Validation**: Utilities for performing cross-validation to assess the 
  robustness of the fitted models across different folds of the data. This ensures 
  that the models generalize well to unseen data.

- **Visualization**: Functions to plot and visualize the relationships between DBH
  and AGB for different species, making it easier to interpret the fitted 
  models and understand the underlying biological patterns.

- **AGB Calculation**: Functions to apply the fitted models for predicting AGB based 
  on diameter measurements. These predictions are crucial for estimating 
  the biomass of trees in various forest management and conservation applications.

These utilities are critical for integrating allometric equations into workflows 
that involve species-specific biomass estimation, providing robust tools for ecological 
research, carbon accounting, and sustainable forest management.
"""

def visualize_allometric_relationship(df, title):
    """
    Visualize the allometric relationship between DBH and above-ground biomass.

    Parameters:
    - df (pd.DataFrame): the input dataframe
    - title (str): the title of the plot

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="diameter", y="AGB", hue="name")
    plt.title(title)
    plt.xlabel("Crown Area (cm)")
    plt.ylabel("Above-Ground Biomass (kg)")
    plt.legend(title="Species")
    plt.show()


def fit_and_evaluate_log_log_model(data, species):
    """
    Fit a log-log model to the data and evaluate its performance.

    Parameters:
    - data (pd.DataFrame): the input dataframe
    - species (str): the species to analyze

    Returns:
    - model: the trained linear regression model
    - intercept: the intercept of the model
    - coefficient: the coefficient of the model
    """
    species_data = data[data["name"] == species]

    # Prepare the data (log transformation)
    X = np.log(species_data[["diameter"]])
    y = np.log(species_data["AGB"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Extract coefficients
    intercept_log = model.intercept_
    coefficient = model.coef_[0]
    intercept = np.exp(intercept_log)  # Convert back to the original scale

    print(f"Species: {species}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Intercept (a): {intercept}")
    print(f"Coefficient (b): {coefficient}")

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.scatter(X_test, y_test, color="blue", label="Actual (log scale)")
    plt.scatter(X_test, y_pred, color="red", label="Predicted (log scale)")
    plt.title(f"{species} - Log(Crown Area) vs. Log(Above-Ground Biomass)")
    plt.xlabel("Log(DBH) (cm)")
    plt.ylabel("Log(Above-Ground Biomass) (kg)")
    plt.legend()
    plt.show()

    return model, intercept, coefficient


def fit_and_evaluate_linear_model(data, species):
    """
    Fit a linear model to the data and evaluate its performance.

    Parameters:
    - data (pd.DataFrame): the input dataframe
    - species (str): the species to analyze

    Returns:
    - model: the trained linear regression model
    - intercept: the intercept of the model
    - coefficient: the coefficient of the model
    """
    # Filter data for the species
    species_data = data[data["name"] == species]

    # Prepare the data
    X = species_data[["diameter"]]
    y = species_data["AGB"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Extract coefficients
    intercept = model.intercept_
    coefficient = model.coef_[0]

    print(f"Species: {species}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Intercept: {intercept}")
    print(f"Coefficient: {coefficient}")

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.title(f"{species} - Linear Model")
    plt.xlabel("DBH (cm)")
    plt.ylabel("Above-Ground Biomass (kg)")
    plt.legend()
    plt.show()

    return model, intercept, coefficient


def fit_and_evaluate_exponential_model(data, species):
    """
    Fit an exponential model to the data and evaluate its performance.

    Parameters:
    - data (pd.DataFrame): the input dataframe
    - species (str): the species to analyze

    Returns:
    - model: the trained linear regression model
    - intercept: the intercept of the model
    - coefficient: the coefficient of the model
    """
    # Filter data for the species
    species_data = data[data["name"] == species]

    # Prepare the data (log transformation)
    X = species_data[["diameter"]]
    y = np.log(species_data["AGB"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_exp = np.exp(y_pred)
    y_test_exp = np.exp(y_test)
    mse = mean_squared_error(y_test_exp, y_pred_exp)
    r2 = r2_score(y_test_exp, y_pred_exp)

    # Extract coefficients
    intercept_log = model.intercept_
    coefficient = model.coef_[0]
    intercept = np.exp(intercept_log)

    print(f"Species: {species}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Intercept (a): {intercept}")
    print(f"Coefficient (b): {coefficient}")

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.scatter(X_test, y_test_exp, color="blue", label="Actual")
    plt.scatter(X_test, y_pred_exp, color="red", label="Predicted")
    plt.title(f"{species} - Exponential Model")
    plt.xlabel("DBH (cm)")
    plt.ylabel("Above-Ground Biomass (kg)")
    plt.legend()
    plt.show()

    return model, intercept, coefficient


def fit_and_evaluate_logarithmic_model(data, species):
    """
    Fit a logarithmic model to the data and evaluate its performance.

    Parameters:
    - data (pd.DataFrame): the input dataframe
    - species (str): the species to analyze

    Returns:
    - model: the trained linear regression model
    - intercept: the intercept of the model
    - coefficient: the coefficient of the model
    """
    # Filter data for the species
    species_data = data[data["name"] == species]

    # Prepare the data
    X = np.log(species_data[["diameter"]])
    y = species_data["AGB"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Extract coefficients
    intercept = model.intercept_
    coefficient = model.coef_[0]

    print(f"Species: {species}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Intercept: {intercept}")
    print(f"Coefficient: {coefficient}")

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.title(f"{species} - Logarithmic Model")
    plt.xlabel("Log(DBH) (cm)")
    plt.ylabel("Above-Ground Biomass (kg)")
    plt.legend()
    plt.show()

    return model, intercept, coefficient


def fit_and_evaluate_polynomial_model(data, species, degree=2):
    """
    Fit a polynomial model to the data and evaluate its performance.

    Parameters:
    - data (pd.DataFrame): the input dataframe
    - species (str): the species to analyze
    - degree (int): the degree of the polynomial model (default is 2)

    Returns:
    - model: the trained polynomial regression model
    - coefficients: the coefficients of the model
    """
    # Filter data for the species
    species_data = data[data["name"] == species]

    # Prepare the data
    X = species_data[["diameter"]]
    y = species_data["AGB"]

    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Species: {species}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.scatter(X_test[:, 1], y_test, color="blue", label="Actual")
    plt.scatter(X_test[:, 1], y_pred, color="red", label="Predicted")
    plt.title(f"{species} - Polynomial Model (Degree {degree})")
    plt.xlabel("DBH (cm)")
    plt.ylabel("Above-Ground Biomass (kg)")
    plt.legend()
    plt.show()

    return model, model.coef_


def fit_and_evaluate_gam_model(data, species):
    """
    Fit a Generalized Additive Model (GAM) to the data and evaluate its performance.

    Parameters:
    - data (pd.DataFrame): the input dataframe
    - species (str): the species to analyze

    Returns:
    - model: the trained GAM model
    """
    # Filter data for the species
    species_data = data[data["name"] == species]

    # Prepare the data
    X = species_data[["diameter"]].values
    y = species_data["AGB"].values

    # Fit the GAM model
    model = LinearGAM(s(0)).fit(X, y)

    # Predict and evaluate
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Species: {species}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.scatter(X, y, color="blue", label="Actual")
    plt.scatter(X, y_pred, color="red", label="Predicted")
    plt.title(f"{species} - Generalized Additive Model")
    plt.xlabel("DBH (cm)")
    plt.ylabel("Above-Ground Biomass (kg)")
    plt.legend()
    plt.show()

    return model


def calculate_cross_val_score(models, df_cleaned):
    '''Calculate the cross-validation R^2 scores for the specified models.
    
    Parameters:
    - models (dict): A dictionary of species and their corresponding models.
    - df_cleaned (pd.DataFrame): The cleaned dataframe.
    
    Returns:
    - scores (array): The array of cross-validation R^2 scores.'''
    for species, model in models.items():
        species_data = df_cleaned[df_cleaned["name"] == species]
        X = np.log(species_data[["diameter"]])
        y = np.log(species_data["AGB"])

        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        print(f"Species: {species}")
        print(f"Cross-Validation R^2 Scores: {scores}")
        print(f"Mean R^2 Score: {scores.mean()}")
    return scores


def calculate_cross_val_score_poly(models, df_cleaned):
    '''Calculate the cross-validation R^2 scores for the specified polynomial models.
    
    Parameters:
    - models (dict): A dictionary of species and their corresponding polynomial models.
    - df_cleaned (pd.DataFrame): The cleaned dataframe.
    
    Returns:
    - scores (array): The array of cross-validation R^2 scores.'''
    for species, model in models.items():
        species_data = df_cleaned[df_cleaned["name"] == species]
        X = species_data[["diameter"]]
        y = species_data["AGB"]

        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        print(f"Species: {species}")
        print(f"Cross-Validation R^2 Scores: {scores}")
        print(f"Mean R^2 Score: {scores.mean()}")
    return scores


def calculate_cross_val_score_gam(models, df_cleaned):
    '''Calculate the cross-validation R^2 scores for the specified GAM models.
    
    Parameters:
    - models (dict): A dictionary of species and their corresponding GAM models.
    - df_cleaned (pd.DataFrame): The cleaned dataframe.
    
    Returns:
    - scores (array): The array of cross-validation R^2 scores.'''
    for species, model in models.items():
        species_data = df_cleaned[df_cleaned["name"] == species]
        X = species_data[["diameter"]]
        y = species_data["AGB"]

        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        print(f"Species: {species}")
        print(f"Cross-Validation R^2 Scores: {scores}")
        print(f"Mean R^2 Score: {scores.mean()}")
    return scores
