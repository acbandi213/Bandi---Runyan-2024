#Utility functions 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

class psychometrics:
    def scale_stim_values(data): #Scale stimulus values for optimized GL
        Stim_val = []
        stim_vals = stats.zscore([-1, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1])
        stim_conditions = [1,5,3,7,8,4,6,2]
        for x in data.index.tolist():
            for cond in range(8):
                if data['condition'][x] == stim_conditions[cond]:
                    stim_value = stim_vals[cond]
                    Stim_val.append(stim_value)
        return Stim_val

    def perc_left(condition, data, left):
        cond_frame = data[data['condition'] == condition]
        if left == 1:
            percent = len(cond_frame[cond_frame['choice'] == 0])/len(cond_frame)
        elif left == 0:
            percent = len(cond_frame[cond_frame['choice'] == 1])/len(cond_frame)
        return percent

    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return (y)

    def fit_sigmoid(y_data):
        xdata = list(range(0,8))
        ydata = y_data

        p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

        popt, pcov = curve_fit(psychometrics.sigmoid, xdata, ydata,p0, method='dogbox', maxfev=20000)

        x = np.linspace(-1, 8)
        y = psychometrics.sigmoid(x, *popt)
        
        return y 
    
class plotting:
    def plot_psychometric_all_data(data):
        fig, axs = plt.subplots(1,1,figsize = (2.5,4))

        right = [] 
        for i in [1,5,3,7,8,4,6,2]: 
            right.append(psychometrics.perc_left(i,data,0))
        axs.scatter(list(range(0,8)), y = right, color='blue')
        axs.plot(np.linspace(0, 7),psychometrics.fit_sigmoid(y_data=right), color='blue')
        axs.set_xticks(list(range(0,8)))
        axs.set_xticklabels(['-90','-60','-30','-15','15','30','60','90'])
        axs.set_yticks([0,.25,.5,.75,1])
        axs.set_ylabel('% R choice', fontsize=12)
        axs.set_title('All animals \n' + 'num_trials = ' + str(len(data)))

    def plot_psychometric_for_animal(data, animal_ids):
        fig, axs = plt.subplots(3,5,figsize = (16,8))

        for animal_num in range(0,len(animal_ids)):
            if animal_num <= 4:
                slice_of_data = data[data['mouse_id'] == animal_ids[animal_num]]
                num_sesh = len(list(set(slice_of_data['sesh_num'])))
                for sesh in range(0,num_sesh):
                    sesh_slice = slice_of_data[slice_of_data['sesh_num'] == sesh]
                    right = [] 
                    for i in [1,5,3,7,8,4,6,2]:
                        right.append(psychometrics.perc_left(i,slice_of_data,0))
                    axs[0,animal_num].plot(np.linspace(0, 7),psychometrics.fit_sigmoid(y_data=right), color='blue', lw=1.5)
                    axs[0,animal_num].scatter(list(range(0,8)), y = right, color='blue')
                    axs[0,animal_num].set_xticks(list(range(0,8)))
                    axs[0,animal_num].set_xticklabels(['-90','-60','-30','-15','15','30','60','90'])
                    axs[0,animal_num].set_yticks([0,.25,.5,.75,1])
                    axs[0,animal_num].set_ylabel('% R choice', fontsize=12)
                    axs[0,animal_num].set_title(animal_ids[animal_num] + '  num_trials = ' + str(len(slice_of_data)))
            elif animal_num > 4 and animal_num <=9 :
                slice_of_data = data[data['mouse_id'] == animal_ids[animal_num]]
                num_sesh = len(list(set(slice_of_data['sesh_num'])))
                for sesh in range(0,num_sesh):
                    sesh_slice = slice_of_data[slice_of_data['sesh_num'] == sesh]
                    right = [] 
                    for i in [1,5,3,7,8,4,6,2]:
                        right.append(psychometrics.perc_left(i,slice_of_data,0))
                    axs[1,animal_num-5].plot(np.linspace(0, 7),psychometrics.fit_sigmoid(y_data=right), color='blue', lw=1.5)
                    axs[1,animal_num-5].scatter(list(range(0,8)), y = right, color='blue')
                    axs[1,animal_num-5].set_xticks(list(range(0,8)))
                    axs[1,animal_num-5].set_xticklabels(['-90','-60','-30','-15','15','30','60','90'])
                    axs[1,animal_num-5].set_yticks([0,.25,.5,.75,1])
                    axs[1,animal_num-5].set_ylabel('% R choice', fontsize=12)
                    axs[1,animal_num-5].set_title(animal_ids[animal_num] + '  num_trials = ' + str(len(slice_of_data))) 
            elif animal_num > 9:
                slice_of_data = data[data['mouse_id'] == animal_ids[animal_num]]
                num_sesh = len(list(set(slice_of_data['sesh_num'])))
                for sesh in range(0,num_sesh):
                    sesh_slice = slice_of_data[slice_of_data['sesh_num'] == sesh]
                    right = [] 
                    for i in [1,5,3,7,8,4,6,2]:
                        right.append(psychometrics.perc_left(i,slice_of_data,0))
                    axs[2,animal_num-10].plot(np.linspace(0, 7),psychometrics.fit_sigmoid(y_data=right), color='blue', lw=1.5)
                    axs[2,animal_num-10].scatter(list(range(0,8)), y = right, color='blue')
                    axs[2,animal_num-10].set_xticks(list(range(0,8)))
                    axs[2,animal_num-10].set_xticklabels(['-90','-60','-30','-15','15','30','60','90'])
                    axs[2,animal_num-10].set_yticks([0,.25,.5,.75,1])
                    axs[2,animal_num-10].set_ylabel('% R choice', fontsize=12)
                    axs[2,animal_num-10].set_title(animal_ids[animal_num] + '  num_trials = ' + str(len(slice_of_data))) 
                
        fig.tight_layout()