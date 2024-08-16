#diameter_utils.py

#necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import allometric_equations as ae
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dropout, BatchNormalization
import xgboost as xgb
import joblib

def add_bbox_columns(df):
    #throw an error if the necessary columns are not present
    try: 
        if 'xmin' not in df.columns or 'xmax' not in df.columns or 'ymin' not in df.columns or 'ymax' not in df.columns:
            raise ValueError('The necessary columns are not present in the dataframe')
        df['bbox_area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])
        df['bbox_diagonal'] = ((df['xmax'] - df['xmin'])**2 + (df['ymax'] - df['ymin'])**2) ** 0.5

        #bbox diameter as either the height or width, whichever is larger
        df['bbox_across'] = np.where(df['xmax'] - df['xmin'] > df['ymax'] - df['ymin'], df['xmax'] - df['xmin'], df['ymax'] - df['ymin'])
    except ValueError as e:
        print(e)
    return df


def remove_outliers(df, column_name):
    #remove outliers; anything outside of 10% and 90% quantiles
    df = df[(df[column_name] > df[column_name].quantile(0.10)) & (df[column_name] < df[column_name].quantile(0.90))]
    return df

def prepare_data(df, column_name, one_hot_encode = True):
    df = add_bbox_columns(df)
    if one_hot_encode is True:
        df = pd.get_dummies(df, columns=['name'], dtype=int)
    df = remove_outliers(df, column_name)
    return df

def split_data(df, scaler):
    #load the scaler 
    if scaler is True: 
        scaler = joblib.load('pkl_files/diameter_scaler.pkl')
    
    selected_columns = ['bbox_area', 'bbox_diagonal', 'bbox_across', 'name_Cacao', 'name_Musacea', 'name_Guaba', 'name_Mango', 'name_Otra variedad', 'diameter']
    if not all(col in df.columns for col in selected_columns):
        raise ValueError('The necessary columns are not present in the dataframe')
    
    X_process = df[['bbox_area', 'bbox_diagonal', 'bbox_across', 
        'name_Cacao', 'name_Musacea', 'name_Guaba', 'name_Mango', 'name_Otra variedad']].values
    y_process = df['diameter'].values

    if scaler is False: 
        scaler = StandardScaler()
        scaler.fit(X_process)
        joblib.dump(scaler, 'pkl_files/diameter_scaler.pkl')

    X_process_scaled = scaler.transform(X_process)

    return X_process_scaled, y_process

def baseline_model (X_train):
    model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
    ])

    model.compile(optimizer=Adam(), loss='mean_squared_error')

    return model

def mse_and_rmse(y_test, y_pred): 
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error: {rmse}')
    return mse, rmse

def plot_results(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(y_test))
    plt.scatter(indices, y_test, color='blue', label='Actual Diameter', alpha=0.6)
    plt.scatter(indices, y_pred, color='red', label='Predicted Diameter', alpha=0.6)
    plt.xlabel('Index')
    plt.ylabel('Diameter (units)')
    plt.title('Scatter Plot of Actual vs. Predicted Tree Crown Diameters')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare(y_test, y_pred, head_or_tail = 'head'):
    df_compare = pd.DataFrame({'Actual Diameter': y_test, 'Predicted Diameter': y_pred})
    df_compare['Difference (Absolute)'] = abs(df_compare['Actual Diameter'] - df_compare['Predicted Diameter'])
    if head_or_tail == 'head':
        result = df_compare.head(20)
    if head_or_tail == 'tail':
        result = df_compare.tail(20)
    print(result)

def SVR_model(X_train, y_train, X_val, y_val):
    # Define a range of hyperparameters for iteration
    C_values = [0.01, 0.1, 1, 10, 100]
    epsilon_values = [0.01, 0.1, 0.5, 1]
    gamma_values = ['scale', 'auto', 0.1, 1]

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

    # Optionally, find the best parameters
    best_params = sorted(results, key=lambda x: x[3])[0]
    print(f"Best parameters: C={best_params[0]}, epsilon={best_params[1]}, gamma={best_params[2]}, MSE={best_params[3]}")

    return best_params

def best_SVR(best_params): 
    model = SVR(C=best_params[0], epsilon=best_params[1], gamma=best_params[2])
    return model

def CNN(X_train):
    model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def xgboost(X_train, y_train, X_val, y_val):
    xg_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=1,
    verbosity=1,
    random_state=42
    )
    
    xg_model.fit(
    X_train,  # Use the scaled training data
    y_train,
    eval_set=[(X_val, y_val)],  # Including validation and optionally test set
    verbose=True
    )

    return xg_model
    