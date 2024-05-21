#Utility functions 
import os 
import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import ssm
from ssm.util import find_permutation
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from scipy.ndimage.filters import gaussian_filter
from general_utilities import *
from HMM_functions import *

class decoders:
    """
    Class containing functions for decoding and analyzing neural data.
    """

    def load_imaging_spk(path):
        """
        Load the imaging_spk data from a MATLAB file.

        Args:
            path (str): Path of the selected imaging data session.

        Returns:
            h5py.File: Compressed h5py imaging_spk file.
        """
        os.chdir(path)
        arrays = {}
        f = h5py.File('imaging_spk.mat')
        for k, v in f.items():
            arrays[k] = np.array(v)
        
        return f 

    def identify_imaged_trials_in_imaging_spk(path):
        """
        Identify which trials in imaging_spk have neural data.

        Args:
            path (str): Path of the selected imaging data session.

        Returns:
            list: List of trials that were imaged (0:n).
        """
        f = decoders.load_imaging_spk(path)
        num_trials = f['imaging_spk']['start_it'].shape[0]
        #Figure out which trials were imaged and which were not 
        trial_info = []
        for trial in range(0,num_trials):
            file = f['imaging_spk']['start_it'][trial][0]
            file_open = np.array(f[file])
            if file_open[0] == 0:
                trial_info.append(0) 
            elif file_open[0] > 0:
                trial_info.append(file_open[0][0]) 
        trial_info = np.array(trial_info, dtype=object)
        imaged_trials = np.where(trial_info > 1)[0].tolist()
        non_imaged_trials = np.where(trial_info < 1)[0].tolist()
        
        return imaged_trials

    def extract_neuraldata_from_imaging_spk(path, event_for_alignment, numFrames):
        """
        Extract neural data from imaging_spk.

        Args:
            path (str): Path of the selected imaging data session.
            event_for_alignment (str): Task event used for alignment. Can be one of:
                'running_start', 'stimulus_on_frames', 'reward_on_frames'.
            numFrames (int): Number of frames in each trial to extract.

        Returns:
            numpy.ndarray: Extracted neural activity from chosen event onset + numFrames for all imaging trials.
                Shape: (trials, neurons, frames)
        """
        f = decoders.load_imaging_spk(path)
        imaged_trials = decoders.identify_imaged_trials_in_imaging_spk(path)

        neuron = f['imaging_spk']['dff_zscored'][imaged_trials[0]][0]
        cell_activity = np.array(f[neuron])
        num_cells = cell_activity.shape[1]

        #Create Matrix of neural data for each cell on a trial (sub sampled)
        all_trials = []
        for trial in imaged_trials:
            neuron = f['imaging_spk']['dff_zscored'][trial][0]
            cell_activity = np.array(f[neuron])[:,:]
            cell_activity = cell_activity.T
            info = f['imaging_spk']['frame_timing_info'][trial][0]
            start = np.array(f[info][event_for_alignment][0][0]).tolist()
            stop = start+numFrames
            if cell_activity.shape[0] > stop:
                rnd = np.round(np.linspace(start, cell_activity.shape[0], numFrames)).astype(int)
            elif cell_activity.shape[0] <= stop:
                rnd = np.round(np.linspace(start, stop, numFrames)).astype(int)
            cell_activity_subsampled = cell_activity[:,rnd]
            all_trials.append(cell_activity_subsampled)
        all_trials = np.array(all_trials, dtype=object)

        return all_trials

    def extract_df_imaging(df, path, path_list, imaging_sesh):
        """
        Extract a DataFrame of trial information for the selected imaging session.

        Args:
            df (pandas.DataFrame): Large DataFrame of trial information across all sessions for all mice.
            path (str): Path of the selected imaging data session.
            path_list (list): List of all paths for all imaging data sessions.
            imaging_sesh (int): Index of the imaging session in path_list.

        Returns:
            pandas.DataFrame: DataFrame of relevant trial information for the selected imaging data session.
        """
        f = decoders.load_imaging_spk(path)
        imaged_trials = decoders.identify_imaged_trials_in_imaging_spk(path)

        mouse_name = path_list[imaging_sesh][-8:-5]
        date = "['14{}']".format(path_list[imaging_sesh][-4:])
        df_slice = df[(df['mouse_id'] == mouse_name) & (df['date'] == date)]
        df_slice = df_slice.reset_index()
        df_slice['new_index'] = df_slice.index.tolist()
        df_imaged = df_slice.loc[df_slice.apply(lambda x: x['new_index'] in imaged_trials, axis=1)]
        df_imaged = df_imaged.reset_index()
        
        return df_imaged
    
    def principal_component_analysis(neural_activity_imaged_trials, n_components):
        """
        Perform principal component analysis on selected imaging session neural data.

        Args:
            neural_activity_imaged_trials (numpy.ndarray): Trialized neural activity from extract_neuraldata_from_imaging_spk.
            n_components (int): Number of principal components.

        Returns:
            tuple: A tuple containing:
                - activity_pc (numpy.ndarray): Dimensionally reduced neural data. Shape: (frames, PC)
                - activity_pc_reshaped (numpy.ndarray): Reshaped dimensionally reduced neural data. Shape: (trials, frames, PC)
        """
        neural_data_trialized_transposed = neural_activity_imaged_trials.transpose(0,2,1)
        neural_data_trialized_reshaped = neural_data_trialized_transposed.reshape(neural_data_trialized_transposed.shape[0]*neural_data_trialized_transposed.shape[1],
                                                                            neural_data_trialized_transposed.shape[2])
        activity = neural_data_trialized_reshaped  
        pca = PCA(n_components)
        pca.fit(activity)  # activity (Time points, Neurons)
        activity_pc = pca.transform(activity)  # transform to low-dimension 
        activity_pc_reshaped = activity_pc.reshape(neural_data_trialized_transposed.shape[0],
                                            neural_data_trialized_transposed.shape[1], 
                                            n_components)
        return activity_pc, activity_pc_reshaped                                                          

    def SVM_linear_decoder(df, neural_activity_imaged_trials, feature):
        """
        Train a linear support vector machine (SVM) for classification of task feature information given neural data.

        Args:
            df (pandas.DataFrame): DataFrame from the selected imaging session.
            neural_activity_imaged_trials (numpy.ndarray): Neural activity from the selected imaging session.
            feature (str): Feature information found in df_imaged. Examples: 'choice', 'correct', 'state'.

        Returns:
            tuple: A tuple containing:
                - cv_scores (numpy.ndarray): 5-fold cross-validation accuracy scores.
                - accuracy (float): Accuracy score on the test data.
                - percent_of_zero_in_y (float): Percentage of zero values in the target variable.
        """
        neural_data = neural_activity_imaged_trials
        neural_data_reshape = neural_data.reshape(neural_data.shape[0], 
                                                neural_data.shape[1]*neural_data.shape[2])

        feature_all_trials_vector = []
        for trial in df.index:
            value = df[feature].iloc[trial]
            feature_all_trials_vector.append(value)
        feature_all_trials_vector = np.array(feature_all_trials_vector)
        if feature != 'choice':
            feature_all_trials_vector[feature_all_trials_vector == 2] = 1
            class_weight = 'balanced'
        elif feature == 'choice':
            feature_all_trials_vector = feature_all_trials_vector
            class_weight = None

        X = neural_data_reshape
        y = feature_all_trials_vector

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  

        model_svm = SVC(random_state=0, kernel='linear', C=1, decision_function_shape='ovr', class_weight=class_weight)
        model_svm.fit(X_train, y_train)
        y_test_pred = model_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        cv_scores = cross_val_score(model_svm, X, y, cv=5)
        percent_of_zero_in_y = len(y[y==0])/len(y)
        
        return cv_scores, accuracy, percent_of_zero_in_y 
    
    def tensor_component_analysis(neural_activity_imaged_trials, rank): 
        """
        Perform non-negative tensor component analysis on the neural activity tensor.

        Args:
            neural_activity_imaged_trials (numpy.ndarray): Neural activity tensor. Shape: (trials, neurons, time)
            rank (int): Rank of the tensor component analysis (i.e., number of factors).

        Returns:
            tuple: A tuple containing:
                - Factor_matrices_rearranged (numpy.ndarray): TCA n-rank matrix. Shape: (neurons, time, factors)
                - reconstructed_tensor (tensorly.tensor): Reconstructed tensor.
        """
        #non negative tensor component analysis on tensor output from extract_imaging_spk
        data = neural_activity_imaged_trials.astype(float)
        # Normalize the data to a non-negative range
        data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Apply smoothing with a Gaussian kernel
        sigma = 1  # Smoothing parameter
        data_smoothed = gaussian_filter(data_normalized, sigma=sigma)

        # Perform NTCA on the preprocessed data
        rank = rank  # Desired rank or number of components

        # Reshape the data into a tensor
        data_tensor = tl.tensor(data_smoothed)

        # Perform Non-Negative Parafac decomposition (NTCA)
        factors = non_negative_parafac(data_tensor, rank=rank)
        reconstructed_tensor = tl.cp_to_tensor(factors)

        # Retrieve the factor matrices
        factor_matrices = factors.factors
        Factor_matrices_rearranged = []
        for factor in [1,2,0]:
            fac_mat_rearrange = factor_matrices[factor][:,:]
            Factor_matrices_rearranged.append(fac_mat_rearrange)
        Factor_matrices_rearranged = np.array(Factor_matrices_rearranged, dtype=object)
        
        return Factor_matrices_rearranged, reconstructed_tensor
    
    def extract_pre_post_turn_neural_data(path, preturn_frame_size, postturn_frame_size):
        """
        Extract pre-turn and post-turn neural data from imaging_spk by alignment to x-position in the maze.

        Args:
            path (str): Path of the selected imaging data session.
            preturn_frame_size (int): Number of frames to extract before the turn.
            postturn_frame_size (int): Number of frames to extract after the turn.

        Returns:
            tuple: A tuple containing:
                - neural_activity_imaged_trials_preturn (numpy.ndarray): Pre-turn neural activity. Shape: (trials, neurons, frames)
                - neural_activity_imaged_trials_postturn (numpy.ndarray): Post-turn neural activity. Shape: (trials, neurons, frames)
        """
        f = decoders.load_imaging_spk(path)
        imaged_trials = decoders.identify_imaged_trials_in_imaging_spk(path)

        neuron = f['imaging_spk']['dff_zscored'][imaged_trials[0]][0]
        cell_activity = np.array(f[neuron])
        num_cells = cell_activity.shape[1]

        neural_activity_imaged_trials_preturn = []
        neural_activity_imaged_trials_postturn = []

        for trial in imaged_trials:
            neuron = f['imaging_spk']['dff_zscored'][trial][0]
            x_position = f['imaging_spk']['x_position'][trial][0]
            cell_activity = np.array(f[neuron])[:,:]
            cell_activity = cell_activity.T
            X_pos = np.array(f[x_position])[:,0]
            rnd = np.round(np.linspace(0, X_pos.shape[0]-1, cell_activity.shape[1])).astype(int)
            X_pos = X_pos[rnd]
            threshold = 1 # Given number to compare the difference
            diff_array = np.abs(np.diff(X_pos)) # Calculate the absolute difference between consecutive elements
            indices = np.where(diff_array > threshold)[0] # Find the indices where the difference is greater than the threshold
            turn_frame = indices[0] #inferred frame of turn 

            if turn_frame+postturn_frame_size > cell_activity.shape[1]:
                rnd = np.round(np.linspace(turn_frame, cell_activity.shape[1]-1, postturn_frame_size)).astype(int)
                cell_activity_postturn = cell_activity[:,rnd]
            elif turn_frame+postturn_frame_size <= cell_activity.shape[1]:
                cell_activity_postturn = cell_activity[:,turn_frame:turn_frame+postturn_frame_size]

            if turn_frame-preturn_frame_size < 0:
                rnd = np.round(np.linspace(0, turn_frame, preturn_frame_size)).astype(int)
                cell_activity_preturn = cell_activity[:,rnd]
            elif turn_frame-preturn_frame_size >= 0:
                cell_activity_preturn = cell_activity[:,turn_frame-preturn_frame_size:turn_frame]

            neural_activity_imaged_trials_preturn.append(cell_activity_preturn)
            neural_activity_imaged_trials_postturn.append(cell_activity_postturn)

        neural_activity_imaged_trials_preturn = np.array(neural_activity_imaged_trials_preturn, dtype=object)
        neural_activity_imaged_trials_postturn = np.array(neural_activity_imaged_trials_postturn, dtype=object)
        
        return neural_activity_imaged_trials_preturn, neural_activity_imaged_trials_postturn

    def SVM_acc_region(state_prob, df, path_list, feature):
        """
        Train SVM classifiers and compute accuracy for a specific feature across multiple imaging sessions.

        Args:
            state_prob (numpy.ndarray): State probabilities.
            df (pandas.DataFrame): DataFrame containing trial information.
            path_list (list): List of paths for all imaging data sessions.
            feature (str): Feature to decode (e.g., 'choice', 'correct', 'state').

        Returns:
            tuple: A tuple containing:
                - SVM_cv_acc_all_sessions (numpy.ndarray): Cross-validation accuracies for all sessions.
                - mean (float): Mean cross-validation accuracy across sessions.
                - err (float): Standard deviation of cross-validation accuracies across sessions.
                - perc_state_1_all_sessions (list): Percentage of trials in state 1 for all sessions.
        """
        Stim = []
        for trial in df.index.tolist():
            if df['stim_value'][trial] < 0:
                stim = 0
            elif df['stim_value'][trial] > 0:
                stim = 1
            Stim.append(stim)
        df['stimLR'] = Stim
        
        preturn_frame_size = 30
        postturn_frame_size = 30
        n_components = 5
        SVM_cv_acc_all_sessions = []
        perc_state_1_all_sessions = []
        for imaging_sesh in list(range(len(path_list))):
            path = path_list[imaging_sesh]
            neural_activity_imaged_trials = decoders.extract_pre_post_turn_neural_data(path, preturn_frame_size, postturn_frame_size)[0]
            #neural_activity_imaged_trials = extract_neuraldata_from_imaging_spk_based_on_event(path, event_for_alignment, numFrames=30)

            PCA = decoders.principal_component_analysis(neural_activity_imaged_trials, n_components)[1]
            df_imaged = decoders.extract_df_imaging(df, path, path_list, imaging_sesh)
            SVM = decoders.SVM_linear_decoder(df_imaged, neural_activity_imaged_trials, feature)
            SVM_cv_acc = SVM[0]
            #print("{} decoding accuracy = {}".format(feature, SVM_cv_acc))
            SVM_cv_acc_all_sessions.append(SVM_cv_acc*100)
            perc_state_1_all_sessions.append(SVM[2]*100)

        SVM_cv_acc_all_sessions = np.array(SVM_cv_acc_all_sessions)
        mean = SVM_cv_acc_all_sessions.mean(axis=1)
        err = SVM_cv_acc_all_sessions.std(axis=1)

        return SVM_cv_acc_all_sessions, mean, err, perc_state_1_all_sessions