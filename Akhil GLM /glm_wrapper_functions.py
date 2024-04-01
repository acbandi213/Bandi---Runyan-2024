import numpy as np
import os 
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import warnings
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import SplineTransformer, QuantileTransformer
from scipy import stats
import scipy.stats
import seaborn as sns
import tensorflow as tf
import scipy.io
from tqdm import tqdm
import pickle
import time

from itertools import accumulate
from collections import Counter

class glm_wrapper_functions:
    def get_data_from_each_CV_fold(train_directory, test_directory, fold):
        directory_train = train_directory.format(fold)
        os.chdir(directory_train)

        behav = scipy.io.loadmat('behav_big_matrix.mat')
        behav_ids = scipy.io.loadmat('behav_big_matrix_ids.mat')

        behav_matrix = behav['behav_big_matrix']
        behav_ids_matrix = behav_ids['behav_big_matrix_ids'][0]

        response = scipy.io.loadmat('combined_response.mat')
        response_matrix = response['combined_response']
        response_matrix[response_matrix > 0.05] = 1

        X_train = behav_matrix
        Y_train = response_matrix
        
        behav_IDS = []
        for trial in list(range(behav_ids['behav_big_matrix_ids'][0].shape[0])):
            behav_IDS.append(behav_ids['behav_big_matrix_ids'][0][trial][0])

        # Count the occurrences of each element in the list
        counter = Counter(behav_IDS)

        # Get the unique values
        unique_values = list(counter.keys())

        # Get the count of each unique value
        count_of_values = list(counter.values())

        IDS_index = np.array(list(accumulate(count_of_values)))-1
        IDS_for_count_of_values = []
        for index in IDS_index[0:418]:
            IDS_for_count_of_values.append(behav_IDS[index])
            
        directory_test = test_directory.format(fold)
        os.chdir(directory_test)

        behav = scipy.io.loadmat('behav_big_matrix.mat')
        behav_ids = scipy.io.loadmat('behav_big_matrix_ids.mat')

        behav_matrix = behav['behav_big_matrix']
        behav_ids_matrix = behav_ids['behav_big_matrix_ids'][0]

        response = scipy.io.loadmat('combined_response.mat')
        response_matrix = response['combined_response']
        response_matrix[response_matrix > 0.05] = 1

        X_test = behav_matrix
        Y_test = response_matrix
        
        # Clean up design matrix and z-score along sample dimension
        X_train = X_train.T
        # Multiply deconvolved activity by 10 to mimic spike number
        Y_train = 10 * Y_train.T

        X_test = X_test.T
        # Multiply deconvolved activity by 10 to mimic spike number
        Y_test = 10 * Y_test.T
        
        return X_train, Y_train, X_test, Y_test, count_of_values, IDS_for_count_of_values

    def train_model_on_cv_fold(X_train, Y_train, count_of_values):
        # Reset keras states
        tf.keras.backend.clear_session()

        model = glm.GLM(activation = 'exp', loss_type = 'poisson', 
                        regularization = 'elastic_net', lambda_series = 10.0 ** np.linspace(-1, -8, 10), 
                        l1_ratio = 0.98, smooth_strength = 0., 
                        optimizer = 'adam', learning_rate = 1e-3)
        
        model.fit(X_train, Y_train, feature_group_size = count_of_values, verbose = False) 
        
        return model

    def train_model_on_cv_fold_GPU(X_train, Y_train, count_of_values):
    # Ensure the function runs on GPU
    with tf.device('/GPU:0'):
        # Reset keras states
        tf.keras.backend.clear_session()

        model = glm.GLM(activation = 'exp', loss_type = 'poisson', 
                        regularization = 'elastic_net', lambda_series = 10.0 ** np.linspace(-1, -8, 10), 
                        l1_ratio = 0.98, smooth_strength = 0., 
                        optimizer = 'adam', learning_rate = 1e-3)
        # Capture the start time
        start_time = time.time()
        
        # Fit the model
        model.fit(X_train, Y_train, feature_group_size=count_of_values, verbose=False) 
        
        # Capture the end time
        end_time = time.time()

    print(f"Time taken on GPU: {end_time - start_time:.4f} seconds")
    
    return model