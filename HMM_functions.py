#Utility functions 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import ssm
from ssm.util import find_permutation
from general_utilities import *

colors = np.array([[39,110,167],[237,177,32],[233,0,111],[0,102,51]])/255

class model_setup:

    def count_sessions(data, animal_ids):
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

    def get_transition_matrix(new_glmhmm):
        recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
        return recovered_trans_mat

    def plot_transition_matrix(recovered_trans_mat, num_states, dpi):
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
        recovered_weights = -new_glmhmm.observations.params
        bias_weights = recovered_weights[:,:,1][:,0].tolist()
        state1_pos = min(range(len(bias_weights)), key=lambda i: abs(bias_weights[i]))
        state2_pos = bias_weights.index(max(bias_weights))
        state3_pos = bias_weights.index(min(bias_weights))
        rearrange_pos = [state1_pos, state2_pos, state3_pos]
        recovered_weights = recovered_weights[rearrange_pos]
        return recovered_weights, rearrange_pos

    def plot_B_weights(recovered_weights, num_states, input_dim, dpi):
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
        posterior_probs_concat = np.concatenate(posterior_probs_rearrange)
        # get state with maximum posterior probability at particular trial:
        state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
        # now obtain state fractional occupancies:
        _, state_occupancies = np.unique(state_max_posterior, return_counts=True)
        state_occupancies = state_occupancies/np.sum(state_occupancies)
        return state_max_posterior, state_occupancies

    def plot_state_occupancy_accuracy_all_data(glm_data, num_states, state_occupancies, dpi):
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



