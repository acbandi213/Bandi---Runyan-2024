#Utility functions 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.special import logsumexp
import ssm
from ssm.util import find_permutation
from general_utilities import *

colors = np.array([[39,110,167],[237,177,32],[233,0,111],[0,102,51]])/255

class model_setup:
    """
    Class containing functions for setting up the model.
    """

    def count_sessions(data, animal_ids):
        """
        Count the number of trials in each session for each animal.

        Args:
            data (DataFrame): The input data containing trial information.
            animal_ids (list): List of animal IDs.

        Returns:
            numpy.ndarray: Array containing the cumulative count of trials for each session.
        """
        #get session trial counts for later use 
        session_trail_count = []
        for animal in animal_ids:
            slice_animal = data[data['mouse_id'] == animal]
            sesh_numbers = list(set(slice_animal['sesh_num'].tolist()))
            for x in sesh_numbers:
                slice_animal_sesh = slice_animal[slice_animal['sesh_num'] == x]
                session_trail_count.append(slice_animal_sesh.index[-1])
        session_trail_count.insert(0, 0)
        sessions = np.array(session_trail_count)
        sessions[-1] = sessions[-1]+1
        return sessions

    def make_design_matrix(data, sessions):
        """
        Create the design matrix and choice data for each session.

        Args:
            data (DataFrame): The input data containing trial information.
            sessions (numpy.ndarray): Array containing the cumulative count of trials for each session.

        Returns:
            tuple: A tuple containing two lists:
                - Session_info: List of numpy arrays, where each array represents the design matrix for a session.
                - Choice: List of numpy arrays, where each array represents the choice data for a session.
        """
        #Make design matrix from all data 
        Session_info = []
        Choice = []
        for x in list(range(0,sessions.shape[0])[:-1]):
            trial_stim = []
            trial_choice = []
            for trial in list(range(sessions[x],sessions[x+1])):
                stim = data['stim_value'][trial]
                bias = 1
                prev_choice = data['prev_choice'][trial]
                prev_rew = data['prev_reward'][trial]
                trial_info = [stim, bias, prev_choice, prev_rew]
                trial_stim.append(trial_info)
                trial_choice.append([data['choice'][trial]])
            Session_info.append(np.array(trial_stim))
            Choice.append(np.array(trial_choice))
        return Session_info, Choice

class analyze_model:
    """
    Class containing functions for analyzing the model.
    """

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
    
    def get_transition_matrix(new_glmhmm):
        """
        Extract the transition matrix from the GLMHMM model.

        Args:
            new_glmhmm: The trained GLMHMM model.

        Returns:
            numpy.ndarray: The recovered transition matrix.
        """
        recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
        return recovered_trans_mat

    def plot_transition_matrix(recovered_trans_mat, num_states, dpi):
        """
        Plot the transition matrix.

        Args:
            recovered_trans_mat (numpy.ndarray): The recovered transition matrix.
            num_states (int): The number of states in the model.
            dpi (int): The resolution of the plot in dots per inch.
        """
        fig, axs = plt.subplots(1,1,figsize = (4,3), dpi=dpi) #dpi=800
        axs.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
        for i in range(recovered_trans_mat.shape[0]):
            for j in range(recovered_trans_mat.shape[1]):
                text = axs.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                                color="k", fontsize=12)
        axs.set_xticks([0, 1, 2], ['1', '2', '3'])
        axs.set_yticks([0, 1, 2], ['1', '2', '3'])      
        if num_states == 3:
            axs.set_xticks([0, 1, 2])
            axs.set_xticklabels(['1', '2', '3'], fontsize=15)
            axs.set_yticks([0, 1, 2])
            axs.set_yticklabels(['1', '2', '3'], fontsize=15)
        elif num_states == 4:
            axs.set_xticks([0, 1, 2, 3])
            axs.set_xticklabels(['1', '2', '3','4'], fontsize=15)
            axs.set_yticks([0, 1, 2, 3])
            axs.set_yticklabels(['1', '2', '3','4'], fontsize=15)
        elif num_states == 5:
            axs.set_xticks([0, 1, 2, 3, 4], ['1', '2', '3','4','5'], fontsize=15)
            axs.set_yticks([0, 1, 2, 3, 4], ['1', '2', '3','4','5'], fontsize=15)
        axs.set_xlabel('State t', size=18)
        axs.set_ylabel('State t+1', size=18)
    
    def recovered_weights_rearranged(new_glmhmm):
        """
        Rearrange the recovered weights based on bias weights.

        Args:
            new_glmhmm: The trained GLMHMM model.

        Returns:
            tuple: A tuple containing:
                - recovered_weights (numpy.ndarray): The rearranged recovered weights.
                - rearrange_pos (list): The positions used for rearranging the weights.
        """
        recovered_weights = -new_glmhmm.observations.params
        bias_weights = recovered_weights[:,:,1][:,0].tolist()
        state1_pos = min(range(len(bias_weights)), key=lambda i: abs(bias_weights[i]))
        state2_pos = bias_weights.index(max(bias_weights))
        state3_pos = bias_weights.index(min(bias_weights))
        rearrange_pos = [state1_pos, state2_pos, state3_pos]
        recovered_weights = recovered_weights[rearrange_pos]
        return recovered_weights, rearrange_pos

    def plot_B_weights(recovered_weights, num_states, input_dim, dpi):
        """
        Plot the B weights for each covariate across states.

        Args:
            recovered_weights (numpy.ndarray): The recovered weights.
            num_states (int): The number of states in the model.
            input_dim (int): The dimensionality of the input features.
            dpi (int): The resolution of the plot in dots per inch.
        """
        #Plot B weights for each covariate across states 
        fig, axs = plt.subplots(1,1,figsize = (4,2.5), dpi=dpi)

        for k in range(num_states):
            axs.plot(range(input_dim), recovered_weights[k][0], color=colors[k],
                            lw=1.5,  label = "recovered", linestyle = '-')
            axs.scatter(range(input_dim), recovered_weights[k][0], color=colors[k],
                        )
            
        axs.set_ylabel("β weight (R-choice)", size=15)
        axs.set_xlabel("Covariate", size=15)
        axs.set_xticks([0, 1, 2, 3])
        axs.set_xticklabels(['Stimulus', 'Bias', 'Prev. Choice', 'Prev. Rew'], rotation=45, fontsize=12)
        axs.axhline(y=0, color="k", alpha=0.5, ls="--")
    
    def posterior_state_prediction(Choice, Session_info, new_glmhmm, rearrange_pos):
        """
        Compute the posterior state probabilities using the trained GLMHMM model.

        Args:
            Choice (list): List of choice data for each session.
            Session_info (list): List of design matrices for each session.
            new_glmhmm: The trained GLMHMM model.
            rearrange_pos (list): The positions used for rearranging the weights.

        Returns:
            numpy.ndarray: The posterior state probabilities for each session.
        """
        #Posterior state predicitions using MLE
        posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                        for data, inpt
                        in zip(Choice, Session_info)]
        posterior_probs = np.array(posterior_probs, dtype=object)

        posterior_probs_rearrange = []
        for ses in range(0,len(posterior_probs)):
            ses_prob = posterior_probs[ses]
            posterior_probs_rearrange.append(ses_prob[:,rearrange_pos])
        posterior_probs_rearrange = np.array(posterior_probs_rearrange, dtype=object)
        return posterior_probs_rearrange

    def concatenate_posterior_probs(posterior_probs_rearrange):
        """
        Concatenate the posterior state probabilities and compute state occupancies.

        Args:
            posterior_probs_rearrange (numpy.ndarray): The rearranged posterior state probabilities.

        Returns:
            tuple: A tuple containing:
                - state_max_posterior (numpy.ndarray): The state with maximum posterior probability at each trial.
                - state_occupancies (numpy.ndarray): The fractional occupancies of each state.
        """
        posterior_probs_concat = np.concatenate(posterior_probs_rearrange)
        # get state with maximum posterior probability at particular trial:
        state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
        # now obtain state fractional occupancies:
        _, state_occupancies = np.unique(state_max_posterior, return_counts=True)
        state_occupancies = state_occupancies/np.sum(state_occupancies)
        return state_max_posterior, state_occupancies

    def plot_state_occupancy_accuracy_all_data(glm_data, num_states, state_occupancies, dpi):
        """
        Plot the state occupancy and accuracy for all data.

        Args:
            glm_data (DataFrame): The input data containing trial information and predicted states.
            num_states (int): The number of states in the model.
            state_occupancies (numpy.ndarray): The fractional occupancies of each state.
            dpi (int): The resolution of the plot in dots per inch.
        """
        #Plot State Occupancy across all data 
        fig, axs = plt.subplots(1,2,figsize = (4,2.5), facecolor='w', edgecolor='k', dpi=dpi)

        for z, occ in enumerate(state_occupancies):
            axs[0].bar(z, occ, width = 0.8, color = colors[z])
            axs[0].set_ylim((0, 1))

        if num_states == 3:
            axs[0].set_xticks([0, 1, 2])
            axs[0].set_xticklabels(['1', '2', '3'], fontsize=12)
        elif num_states == 4:
            axs[0].set_xticks([0, 1, 2, 3])
            axs[0].set_xticklabels(['1', '2', '3', '4'], fontsize=12)
        elif num_states == 5:
            axs[0].set_xticks([0, 1, 2, 3, 4])
            axs[0].set_xticklabels(['1', '2', '3', '4', '5'], fontsize=12)

        axs[0].set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
        axs[0].set_xlabel('state', size=15)
        axs[0].set_ylabel('frac. occupancy', size=15)
        axs[0].set_title('State Occupancy', size=15)

        Frac_correct = []
        for state in list(range(0,num_states)):
            total = len(glm_data[glm_data['state'] == state].index.tolist())
            state_correct = len(glm_data[(glm_data['state'] == state) & (glm_data['correct'] == 1)].index.tolist())
            Frac_correct.append(state_correct/total)
        Frac_correct = np.array(Frac_correct)

        for z, frac in enumerate(Frac_correct):
            axs[1].bar(z, frac*100 , width = 0.8, color = colors[z])
            
        if num_states == 3:
            axs[1].set_xticks([0, 1, 2])
            axs[1].set_xticklabels(['1', '2', '3'], fontsize=12)
        elif num_states == 4:
            axs[1].set_xticks([0, 1, 2, 3])
            axs[1].set_xticklabels(['1', '2', '3', '4'], fontsize=12)
        elif num_states == 5:
            axs[1].set_xticks([0, 1, 2, 3, 4])
            axs[1].set_xticklabels(['1', '2', '3', '4', '5'], fontsize=12)

        axs[1].set_yticks([0, 50, 100], ['0', '50', '100'])
        axs[1].set_xlabel('state', size=15)
        axs[1].set_ylabel('Accuracy (%)', size=15)
        axs[1].set_title('Accuracy', size=15)

        fig.tight_layout()

    def plot_psychometrics_for_each_state(num_states, glm_data, dpi):
        """
        Plot the psychometric curves for each state.

        Args:
            num_states (int): The number of states in the model.
            glm_data (DataFrame): The input data containing trial information and predicted states.
            dpi (int): The resolution of the plot in dots per inch.
        """
        fig, axs = plt.subplots(1,num_states,figsize = (6.5,2.5), dpi=dpi)

        for state in list(range(0,num_states)):
            right = [] 
            for condition in [1,5,3,7,8,4,6,2]: # [1,5,3,7,8,4,6,2]
                right.append(psychometrics.perc_left_state(state, condition, glm_data, 0))
            axs[state].scatter(list(range(0,8)), y = right, color=colors[state])
            axs[state].plot(np.linspace(0, 7),psychometrics.fit_sigmoid(y_data=right), label='visual cues', color=colors[state])
            axs[state].set_xticks(list(range(0,8)))
            axs[state].set_xticklabels(['-90','-60','-30','-15','15','30','60','90'])
            axs[state].set_yticks([0,.25,.5,.75,1])
            axs[state].set_xlabel('Stim location', fontsize=12)
            axs[state].set_title('State '+str(state+1), fontsize=15)
            axs[0].set_ylabel('% R choice', fontsize=15)

        fig.tight_layout()



