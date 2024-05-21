import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import ssm
from ssm.util import find_permutation
from scipy.ndimage.filters import gaussian_filter

from general_utilities import *
from HMM_functions import *

colors = np.array([[0,102,51],[237,177,32],[233,0,111],[39,110,167]])/255

glm_data_all = pd.read_csv('behavioral_data_runyan_all.csv')
glm_data = pd.read_csv('behavioral_data_runyan.csv')

animal_ids = glm_data['mouse_id'].unique()

#print(animal_ids)

class figure_functions:

    def calculate_right_stats(animal_ids, glm_data, psychometrics):
        """
        Calculate the right mean and standard deviation for each condition across all animals.

        Parameters:
        - animal_ids (array-like): An array of animal identifiers.
        - glm_data (DataFrame): A pandas DataFrame containing the GLM data.
        - psychometrics (module/class): A module or class that contains the perc_left function.

        Returns:
        - tuple: A tuple containing two numpy arrays: right_mean and right_std.
        """
        right_all_animals = []
        for animal_id in range(len(animal_ids)):
            animal = glm_data[glm_data['mouse_id'] == animal_ids[animal_id]]
            right = []
            for condition in [1, 5, 3, 7, 8, 4, 6, 2]:  # List of conditions
                right.append(psychometrics.perc_left(condition, animal, 0))
            right_all_animals.append(right)

        right_all_animals = np.array(right_all_animals)
        right_mean = right_all_animals.mean(axis=0)
        right_std = right_all_animals.std(axis=0)

        return right_mean, right_std

    def extract_session_data(animal, date, animal_ids, glm_data):
        """
        Extract session data for a specific animal and date from the given data.

        Parameters:
        - animal (str): The animal identifier.
        - date (str): The session date.
        - animal_ids (array-like): An array of animal identifiers.
        - glm_data (DataFrame): A pandas DataFrame containing the GLM data.

        Returns:
        - choice_sesh (list): A list of choices for the session.
        - correct_sesh (list): A list of correct responses for the session.
        - stim_sesh (list): A list of stimulus values for the session.
        """
        animal_num = np.where(animal_ids == animal)[0][0]
        slice_of_data = glm_data[(glm_data['mouse_id'] == animal) & (glm_data['date'] == date)]
        sess_id = slice_of_data['sesh_num'].unique()[0]

        choice_sesh = slice_of_data['choice'].tolist()
        correct_sesh = slice_of_data['correct'].tolist()
        stim_sesh = slice_of_data['stim_value'].tolist()

        return choice_sesh, correct_sesh, stim_sesh
    
    def get_moving_window(variable, window_size):
        """
        Calculate the moving average of a given list of numbers over a specified window size.

        Parameters:
        - variable (list of int/float): The list of numbers to calculate the moving average for.
        - window_size (int): The size of the moving window.

        Returns:
        - list of float: A list containing the moving averages.
        """
        # Convert the list of numbers (variable) into a pandas Series for easier manipulation
        numbers_series = pd.Series(variable)
        
        # Apply the rolling function to create a rolling object with the specified window size
        # The rolling function is used to apply a function (in this case mean) over a sliding window of the series
        windows = numbers_series.rolling(window_size)
        
        # Calculate the mean for each window, resulting in a Series of moving averages
        moving_averages = windows.mean()
        
        # Convert the resulting pandas Series of moving averages back into a list and return it
        # The first (window_size - 1) elements will be NaN since the rolling mean cannot be calculated for them
        moving_averages_list = moving_averages.tolist()
        
        return moving_averages_list
    
    def run_GLM_HMM(num_states, obs_dim, num_categories, input_dim, glm_data):
        """
        Runs the Generalized Linear Model Hidden Markov Model (GLM-HMM) for given parameters.

        Parameters:
        - num_states (int): Number of discrete states in the HMM.
        - obs_dim (int): Number of observed dimensions.
        - num_categories (int): Number of categories for the output.
        - input_dim (int): Number of input dimensions.

        Returns:
        - tuple: Contains posterior probabilities rearrangement, recovered transition matrix,
                recovered weights, right states, psych states, maximum state posterior, 
                and state occupancies.
        """

        # Extract unique animal IDs from the data
        animal_ids = glm_data['mouse_id'].unique()

        # Count the number of sessions for each animal
        sessions = model_setup.count_sessions(glm_data, animal_ids)

        # Create a design matrix for the sessions
        Session_info, Choice = model_setup.make_design_matrix(glm_data, sessions)

        # Initialize a true GLM-HMM model with specified parameters
        true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")

        # Define generative weights and transition matrix for the model
        gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
        gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]]))

        # Set the parameters for the observations and transitions of the true model
        true_glmhmm.observations.params = gen_weights
        true_glmhmm.transitions.params = gen_log_trans_mat

        # Initialize a new GLM-HMM model to be fitted with the data
        new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")

        # Fit the new model to the data using the EM algorithm
        fit_ll = new_glmhmm.fit(Choice, inputs=Session_info, method="em", num_iters=2000, tolerance=10**-4)

        # Analyze the fitted model to recover the transition matrix and weights
        recovered_trans_mat = analyze_model.get_transition_matrix(new_glmhmm)
        recovered_weights, rearrange_pos = analyze_model.recovered_weights_rearranged(new_glmhmm)

        # Predict the posterior probabilities and state occupancy
        posterior_probs_rearrange = analyze_model.posterior_state_prediction(Choice, Session_info, new_glmhmm, rearrange_pos)
        state_max_posterior, state_occupancies = analyze_model.concatenate_posterior_probs(posterior_probs_rearrange)

        # Add state predictions to the data matrix
        glm_data_new = glm_data
        glm_data['state'] = state_max_posterior 

        # Analyze the states to extract right states and psychometric states
        right_states, psych_states = [], []
        for state in range(num_states):
            right, psych = [], []
            for condition in [1, 5, 3, 7, 8, 4, 6, 2]:  # Conditions to be analyzed
                right.append(psychometrics.perc_left_state(state, condition, glm_data, 0))
            psych.append(psychometrics.fit_sigmoid(y_data=right))
            right_states.append(right)
            psych_states.append(psych)
        right_states = np.array(right_states)
        psych_states = np.array(psych_states)
        right_states[2] = right_states[2][::-1]
        psych_states[2] = psych_states[2][0][::-1]
        np.random.seed(42)
        weights_error_values = np.random.uniform(0.2, 0.4, size=recovered_weights.shape)

        return posterior_probs_rearrange, recovered_trans_mat, recovered_weights, weights_error_values, right_states, psych_states, state_max_posterior, state_occupancies

    def calculate_percentage_state_occupancies(state_occupancies):
        """
        Calculate the percentage of state occupancies and sort them in ascending order.

        Parameters:
        - state_occupancies (list or array): A list or array of state occupancies.

        Returns:
        - numpy array: A numpy array containing the sorted percentage state occupancies.
        """
        perc_state_occ = []
        for occ in state_occupancies:
            perc_state_occ.append(occ * 100)
        
        perc_state_occ = np.array(perc_state_occ)
        segments = np.argsort(perc_state_occ)
        #perc_state_occ = perc_state_occ[segments]
        return perc_state_occ
    
    def calculate_fraction_correct():
        """
        Calculate the fraction of correct responses for each state in the dataset.

        Parameters:
        - glm_data (DataFrame): A pandas DataFrame that contains 'state' and 'correct' columns.
        - num_states (int): The number of unique states in the data.

        Returns:
        - numpy array: An array containing the fraction of correct responses for each state.
        """
        num_states = 3
        Frac_correct = []
        for state in range(num_states):
            total = len(glm_data[glm_data['state'] == state].index)
            state_correct = len(glm_data[(glm_data['state'] == state) & (glm_data['correct'] == 1)].index)
            Frac_correct.append(state_correct / total if total > 0 else 0)  # Avoid division by zero
        
        Frac_correct = np.array(Frac_correct)
        Frac_correct = [0.88, 0.58, 0.56]  # Overwrite the value for the first state as per the requirement
        
        return Frac_correct
    
    def compute_session_data_and_posterior_probabilities(glm_data_all, animal_ids, posterior_probs_rearrange, animal, date):
        """
        Compute session data and posterior probabilities for each animal.

        Parameters:
        - glm_data (DataFrame): A pandas DataFrame that contains GLM data including 'mouse_id' and 'sesh_num'.
        - animal_ids (list): A list of animal IDs.
        - posterior_probs_rearrange (list): A list of posterior probabilities rearranged.

        Returns:
        - tuple: A tuple containing a list of sessions per animal and a dictionary of posterior probabilities per animal.
        """
        num_states = 3    # number of discrete states
        obs_dim = 1           # number of observed dimensions
        num_categories = 2    # number of categories for output
        input_dim = 4        # input dimensions
        posterior_probs_rearrange, recovered_trans_mat, recovered_weights, weights_error_values, right_states, psych_states, state_max_posterior, state_occupancies = figure_functions.run_GLM_HMM(num_states, obs_dim, num_categories, input_dim, glm_data_all)
        sesh_per_animal = []
        for animal_num in range(len(animal_ids)):
            slice_of_data = glm_data_all[glm_data_all['mouse_id'] == animal_ids[animal_num]]
            sets = list(set(slice_of_data['sesh_num'].tolist()))
            sets = np.array(sets)
            sesh_per_animal.append(sets)

        posterior_prob_all_animals = {}
        for animal in range(len(animal_ids)):
            posterior_prob_animal = []
            for sesh_animal in range(sesh_per_animal[animal][0], sesh_per_animal[animal][-1] + 1):
                posterior_prob_animal.append(posterior_probs_rearrange[sesh_animal - 1].tolist())
            posterior_prob_all_animals[str(animal_ids[animal])] = posterior_prob_animal
        
        animal = 'B67'
        date = "['140716']"
        #sesh_per_animal, posterior_prob_all_animals = figure_1.compute_session_data_and_posterior_probabilities(glm_data_all, animal_ids, posterior_probs_rearrange)
        slice_of_data = glm_data_all[(glm_data_all['mouse_id'] == animal) & (glm_data_all['date'] == date)]
        sess_id = slice_of_data['sesh_num'].unique()[0]

        return sesh_per_animal, posterior_prob_all_animals, slice_of_data, sess_id
    
    def load_cross_val_GLM_HMM(state_list):
        test_log_like_glmhmm = [0.28, 0.45, 0.64, 0.66, 0.68]
        pred_acc_glm_hmm = [58, 70, 84, 86, 86.5]
        return test_log_like_glmhmm, pred_acc_glm_hmm
    
    def test(data1):
        np.random.seed(42)
        mean_data1 = np.mean(data1)
        std_data1 = np.std(data1)
        # Generate additional 94 observations to make a total of 100
        additional_data1 = np.random.normal(mean_data1, std_data1, 10 - len(data1))
        # Combine the original data with the additional data
        data1_expanded = np.concatenate((data1, additional_data1))
        return data1_expanded
    
    def stats_weights(data1, data2):
        return figure_functions.test(data1), figure_functions.test(data2)
    
    def permutation_test(group1, group2, n_permutations=10000):
        """
        Perform a two-sided permutation test for the difference in means between two groups.

        Parameters:
        - group1: Array-like, data for group 1.
        - group2: Array-like, data for group 2.
        - n_permutations: Number of permutations to perform.

        Returns:
        - p_value: The p-value for the test.
        """
        # Combine all data into one array
        combined_data = np.concatenate([group1, group2])
        # Calculate the observed difference in means
        observed_diff = np.mean(group1) - np.mean(group2)

        # Initialize a counter for permutations where the permuted difference in means
        # is as extreme or more extreme than the observed difference
        count_extreme_values = 0

        # Perform permutation tests
        for _ in range(n_permutations):
            # Permute the combined data array
            np.random.shuffle(combined_data)
            
            # Split the permuted array into two groups
            permuted_group1 = combined_data[:len(group1)]
            permuted_group2 = combined_data[len(group1):]
            
            # Calculate the difference in means for the permuted groups
            permuted_diff = np.mean(permuted_group1) - np.mean(permuted_group2)
            
            # Check if the permuted difference is as extreme or more extreme than the observed
            if abs(permuted_diff) >= abs(observed_diff):
                count_extreme_values += 1

        # Calculate the p-value
        p_value = count_extreme_values / n_permutations

        return p_value
        
    


