import numpy as np
import pandas as pd
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt
import ssm
from ssm.util import find_permutation
import scipy.io
from scipy.io import loadmat
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

from general_utilities import *
from HMM_functions import *

class data_extraction:

    def create_matrix_aligned_data(mat_data, session_number):
        """
        Create a dictionary containing aligned data for a specific session.

        :param mat_data: MATLAB data containing aligned data.
        :param session_number: The session number to extract data from.
        :return: A dictionary containing aligned data for the specified session.
        """
        aligned_data = mat_data['aligned_data'][0]
        session_data = aligned_data[session_number]
        session = {}
        session['ID'] = session_data[0][0]
        session['is_left_choice'] = session_data[1]
        session['is_left_stimulus'] = session_data[2]
        session['r_aligned_stimulus'] = session_data[3]
        session['x_pos_aligned_stimulus'] = session_data[4]
        session['y_pos_aligned_stimulus'] = session_data[5]
        session['r_aligned_choice'] = session_data[6]
        session['x_pos_aligned_choice'] = session_data[7]
        session['y_pos_aligned_choice'] = session_data[8]
        session['stimulus_frame'] = session_data[9]
        session['choice_frame'] = session_data[10]
        session['sound_location'] = session_data[11]

        return session
    
    def convert_stim_values(arr):
        """
        Convert and z-score stimulus values.

        :param arr: Array of stimulus values.
        :return: A tuple containing the reordered stimulus values and z-scored stimulus values.
        """
        # Mapping of original values to their new values
        value_map = {1: 1, 2: 5, 3: 3, 4: 7, 5: 8, 6: 4, 7: 6, 8: 2}
        # Converting each element in the array
        stims_reordered = [value_map[x] for x in arr]
        stim_vals = stats.zscore([-1, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1])
        Stim_val_zscored = []
        for x in arr:
            stim_val_zscored = stim_vals[x-1]
            Stim_val_zscored.append(stim_val_zscored)
        return np.array(stims_reordered), np.array(Stim_val_zscored)
    
    def create_data_matrix_session(session, Stim_val_zscored):
        """
        Create a data matrix and choice vector for a session.

        :param session: Dictionary containing session data.
        :param Stim_val_zscored: Z-scored stimulus values.
        :return: A tuple containing the data matrix and choice vector for the session.
        """
        data_matrix_session = []
        choice_session = []
        for trial in list(range(len(session['is_left_choice'][:,0]))):
            stim_zscore = Stim_val_zscored[trial]
            bias = 1
            if trial > 0:
                prev_choice = session['is_left_choice'][:,0][trial-1]
                if session['is_left_choice'][:,0][trial-1] == session['is_left_stimulus'][:,0][trial-1]:
                    prev_reward = 1 
                elif session['is_left_choice'][:,0][trial-1] != session['is_left_stimulus'][:,0][trial-1]:
                    prev_reward = 0
            elif trial == 0:
                prev_choice = 0
                prev_reward = 0
            data_matrix_session.append([stim_zscore, bias, prev_choice, prev_reward])
            choice = session['is_left_choice'][:,0][trial]
            if choice == 1:
                choice_change = 0
            elif choice == 0:
                choice_change = 1
            choice_session.append([choice_change])

        return np.array(data_matrix_session), np.array(choice_session)
    
    def balance_trials(data, states, stimulus, choice):
        """
        Balance trials based on state and trial conditions.

        This function selects an equal number of state 1 trials and non-state 1 trials
        (combining state 2 and 3), and then balances them to have a structured distribution
        of trials from each stimulus and choice combination.

        :param data: The dataset containing the neural activity data.
        :param states: An array indicating the state (1, 2, or 3) for each trial.
        :param stimulus: An array indicating the stimulus (0 for left, 1 for right) for each trial.
        :param choice: An array indicating the choice (0 for left, 1 for right) for each trial.
        :return: The balanced dataset with an equal number of state 1 and non-state 1 trials,
                 and a structured distribution of trials from each stimulus and choice combination.
        """
        # Separate trials based on the state
        state_1_trials = data[states == 1]
        state_2_3_trials = data[(states == 2) | (states == 3)]
        
        # Determine the minimum number of trials between state 1 and non-state 1 trials
        min_trials = min(len(state_1_trials), len(state_2_3_trials))
        
        # Create labels for balancing trial conditions
        state_1_labels = (2 * stimulus[states == 1] + choice[states == 1]).astype(str)
        state_2_3_labels = (2 * stimulus[(states == 2) | (states == 3)] + choice[(states == 2) | (states == 3)]).astype(str)
        
        # Initialize empty arrays to store the balanced datasets
        balanced_state_1 = np.empty((0, state_1_trials.shape[1]))
        balanced_state_2_3 = np.empty((0, state_2_3_trials.shape[1]))
        
        # Iterate over each trial condition
        for condition in ['00', '01', '10', '11']:
            # Select trials for the current condition in state 1
            state_1_condition_trials = state_1_trials[state_1_labels == condition]
            state_1_condition_count = len(state_1_condition_trials)
            
            # Select trials for the current condition in non-state 1
            state_2_3_condition_trials = state_2_3_trials[state_2_3_labels == condition]
            state_2_3_condition_count = len(state_2_3_condition_trials)
            
            # Determine the minimum number of trials for the current condition
            min_condition_trials = min(state_1_condition_count, state_2_3_condition_count, min_trials // 4)
            
            # Randomly select an equal number of trials for the current condition
            selected_state_1_trials = state_1_condition_trials[np.random.choice(state_1_condition_count, min_condition_trials, replace=False)]
            selected_state_2_3_trials = state_2_3_condition_trials[np.random.choice(state_2_3_condition_count, min_condition_trials, replace=False)]
            
            # Append the selected trials for the current condition to the balanced datasets
            balanced_state_1 = np.vstack((balanced_state_1, selected_state_1_trials))
            balanced_state_2_3 = np.vstack((balanced_state_2_3, selected_state_2_3_trials))
        
        # Combine the balanced state 1 and non-state 1 datasets
        balanced_data = np.vstack((balanced_state_1, balanced_state_2_3))
        
        # Return the balanced dataset
        return balanced_data
        
    def run_single_glmhmm(glm_data, Choice, Session_info, num_states, obs_dim, num_categories, input_dim):
        """
        Run a single GLM-HMM model.

        :param glm_data: GLM data.
        :param Choice: Choice data.
        :param Session_info: Session information.
        :param num_states: Number of states in the model.
        :param obs_dim: Observation dimension.
        :param num_categories: Number of categories.
        :param input_dim: Input dimension.
        :return: A tuple containing the fitted log-likelihood, recovered transition matrix, recovered weights,
                 rearrange positions, posterior probabilities, state with maximum posterior, state occupancies,
                 and updated GLM data.
        """
        true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                    observation_kwargs=dict(C=num_categories), transitions="standard")

        gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
        gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]]))
        true_glmhmm.observations.params = gen_weights
        true_glmhmm.transitions.params = gen_log_trans_mat

        new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                        observation_kwargs=dict(C=num_categories), transitions="standard")

        N_iters = 2000 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
        fit_ll = new_glmhmm.fit(Choice, inputs=Session_info, method="em", num_iters=N_iters, tolerance=10**-4)

        recovered_trans_mat = analyze_model.get_transition_matrix(new_glmhmm)

        recovered_weights = -new_glmhmm.observations.params
        recovered_weights, rearrange_pos = analyze_model.recovered_weights_rearranged(new_glmhmm)

        posterior_probs_rearrange = analyze_model.posterior_state_prediction(Choice, Session_info, new_glmhmm, rearrange_pos)
        state_max_posterior, state_occupancies = analyze_model.concatenate_posterior_probs(posterior_probs_rearrange)
        glm_data_new = glm_data

        return fit_ll, recovered_trans_mat, recovered_weights, rearrange_pos, posterior_probs_rearrange, state_max_posterior, state_occupancies, glm_data_new
    
    def restructure_states(states):
        """
        Restructure the state array for binary classification: state 0 vs not state 0.

        :param states: Original state array.
        :return: Restructured state array.
        """
        binary_states = np.where(states == 0, 0, 1)
        return binary_states
    
    def decode_states(neural_data, states, n_timepoints):
        """
        Decode states from neural data using a linear SVM and cross-validation.

        :param neural_data: Neural data in the shape (trials, neurons, timepoints).
        :param states: The states for each trial.
        :param n_timepoints: Number of timepoints.
        :return: A list of mean accuracy scores for each timepoint.
        """
        scores = []

        for t in range(n_timepoints):
            # Reshape neural data for timepoint t: (trials, neurons)
            data_t = neural_data[:, :, t]

            # Initialize a linear SVM
            svm = SVC(random_state=0, kernel='linear', C=1, decision_function_shape='ovr')

            # Perform cross-validation and store the mean accuracy
            score = cross_val_score(svm, data_t, states, cv=3).mean()
            scores.append(score)

        return scores
    
    def decoder(neural_data, y):
        """
        Decode states or choices from neural data using a support vector machine (SVM) with cross-validation.

        :param neural_data: Neural data tensor with shape (trials, neurons, time_points).
        :param y: Target variable (states or choices) for each trial.
        :return: A tuple containing the mean accuracies and standard deviations across iterations for each time point.
        """
        n_trials, n_neurons, n_time_points = neural_data.shape

        iterations = 10

        # Initialize a 3D array to store accuracy: iterations x time_points
        all_accuracies = np.zeros((iterations, n_time_points))

        for iteration in range(iterations):
            for time_point in range(n_time_points):
                # Select data at the current time point
                current_time_data = neural_data[:, :, time_point]
                
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(current_time_data, y, test_size=0.3, random_state=iteration)  # Change random_state to iteration for variability
                
                # Train the SVM model
                svm_model = SVC(C=100, gamma=0.1, kernel='rbf')  # Parameters found via grid search 
                svm_model.fit(X_train, y_train)
                
                # Perform cross-validation and store the mean accuracy
                all_accuracies[iteration, time_point] = cross_val_score(svm_model, current_time_data, y, cv=5).mean()

        # Calculate mean and standard deviation across iterations for each time point
        mean_accuracies = np.mean(all_accuracies, axis=0)
        std_accuracies = np.std(all_accuracies, axis=0)

        return mean_accuracies, std_accuracies
    
    def decode_stim_or_choice(session, states, min_num_cells, n_timepoints, alignment_mode, decoding_var, balanced_index):
        """
        Decode stimulus or choice from neural data using an SVM classifier.

        :param session: Session data dictionary.
        :param states: Array of states for each trial.
        :param min_num_cells: Minimum number of cells to use for decoding.
        :param n_timepoints: Number of time points to consider.
        :param alignment_mode: Alignment mode ('r_aligned_choice' or 'r_aligned_stimulus').
        :param decoding_var: Variable to decode ('is_left_choice' or 'is_left_stimulus').
        :param balanced_index: Indices of balanced trials.
        :return: A tuple containing mean accuracies and standard deviations for state 1 trials and all trials.
        """
        neural_data = session[alignment_mode] #r_aligned_choice | r_aligned_stimulus

        y = session[decoding_var][:,0] #is_left_choice | is_left_stimulus

        num_cells_to_use = np.random.randint(0, neural_data.shape[1], min_num_cells)
        neural_data = neural_data[:,num_cells_to_use,:n_timepoints]

        non_state1_trial_index = np.where(states == 0)[0]
        subsample = np.random.randint(0, len(y), len(non_state1_trial_index))
        balanced_index = balanced_index

        mean_accuracies_state1, std_accuracies_state1 = data_extraction.decoder(neural_data[non_state1_trial_index], y[non_state1_trial_index])
        mean_accuracies_all_state, std_accuracies_all_state = data_extraction.decoder(neural_data[balanced_index], y[balanced_index])

        return mean_accuracies_state1, std_accuracies_state1, mean_accuracies_all_state, std_accuracies_all_state
    
    def decode_state(session, states, min_num_cells, n_timepoints, alignment_mode):
        """
        Decode states from neural data using an SVM classifier.

        :param session: Session data dictionary.
        :param states: Array of states for each trial.
        :param min_num_cells: Minimum number of cells to use for decoding.
        :param n_timepoints: Number of time points to consider.
        :param alignment_mode: Alignment mode ('r_aligned_choice' or 'r_aligned_stimulus').
        :return: A tuple containing mean accuracies and standard deviations for decoding states.
        """
        neural_data = session[alignment_mode] #r_aligned_choice | r_aligned_stimulus

        y = states #is_left_choice | is_left_stimulus

        num_cells_to_use = np.random.randint(0, neural_data.shape[1], min_num_cells)
        neural_data = neural_data[:,num_cells_to_use,:n_timepoints]

        mean_accuracies_state, std_accuracies_state = data_extraction.decoder(neural_data, y)

        return mean_accuracies_state, std_accuracies_state

    def decode_choice(session, states):
        """
        Decode choices from neural data using an SVM classifier.

        :param session: Session data dictionary.
        :param states: Array of states for each trial.
        :return: A tuple containing decoding results for state 1 trials and all trials.
        """
        neural_data = session['r_aligned_choice'] #r_aligned_choice | r_aligned_stimulus
        y = session['is_left_choice'][:,0] #is_left_choice | is_left_stimulus
        index = np.where(states == 0)[0]
        subsample = np.random.randint(0, len(y), len(index))

        n_timepoints = 90
        results_state1_choice = data_extraction.decode_states(neural_data[index], y[index], n_timepoints)
        results_all_trials_choice = data_extraction.decode_states(neural_data[subsample], y[subsample], n_timepoints)

        return results_state1_choice, results_all_trials_choice

    def trial_correctness(choices, stimuli):
        """
        Determine the correctness of each trial based on choices and stimuli.

        :param choices: Array of choices for each trial.
        :param stimuli: Array of stimuli for each trial.
        :return: Array of correctness values (1 for correct, 0 for incorrect).
        """
        # Using list comprehension to compare each element
        return [1 if choice == stimulus else 0 for choice, stimulus in zip(choices, stimuli)]

    def subsample_trials(choices, stimuli):
        """
        Subsample trials to balance correct and incorrect trials for each choice category.

        :param choices: Array of choices for each trial.
        :param stimuli: Array of stimuli for each trial.
        :return: Array of indices of the subsampled trials.
        """
        # First, determine the correctness of each trial
        correctness = data_extraction.trial_correctness(choices, stimuli)

        # Initialize lists to store indices of each category
        correct_left = []
        correct_right = []
        incorrect_left = []
        incorrect_right = []

        # Classify each trial
        for idx, (choice, correct) in enumerate(zip(choices, correctness)):
            if correct:
                if choice == 0: # Left
                    correct_left.append(idx)
                else: # Right
                    correct_right.append(idx)
            else:
                if choice == 0: # Left
                    incorrect_left.append(idx)
                else: # Right
                    incorrect_right.append(idx)

        # Determine the minimum number of trials in any category
        min_trials = min(len(correct_left), len(correct_right), len(incorrect_left), len(incorrect_right))

        # Randomly select min_trials from each category
        selected_correct_left = random.sample(correct_left, min_trials)
        selected_correct_right = random.sample(correct_right, min_trials)
        selected_incorrect_left = random.sample(incorrect_left, min_trials)
        selected_incorrect_right = random.sample(incorrect_right, min_trials)

        concatenated_trials = selected_correct_left + selected_correct_right + selected_incorrect_left + selected_incorrect_right

        # Return the selected trials
        return concatenated_trials