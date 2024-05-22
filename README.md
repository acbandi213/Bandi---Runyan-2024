# Code to generate all of the figures in Bandi & Runyan 2024 

## Figure 1 (+ Supp Fig 1 & 2) 

To fit GLM-HMM (global + individual) on Runyan_2017 behavioral data: load behavioral_data_runyan.csv -> use HMM_functions.py + HMM_utilities.py (mainly for plotting) 

To recreate all plots for the figure: Figure 1.ipynb 

## Figure 2 

To train and test SVM decoders: load imaging_spk from Runyan_2017 data -> data_extraction_for_SVM.py + decoder_functions.py (general decoder functions) 

Processed data + decoding results used for figure: decoding_dict.pkl, state_decoding.pkl, stim_choice_decoding.pkl 

To recreate all plots for the figure: Figure 2.ipynb 

## Figures 3 & 4 (+ Supp Fig 3 & 4) 

To load neural data + make train/test folds for GLM: load imaging_spk from Runyan_2017 data -> make_big_data_matrix.m 

To train and test GLM encoding models: Akhil GLM folder (fork of Tseng GLM 2022) -> encoding_GLM.py -> Fit Encoding GLM.ipynb 

Processed data + encoding results used for figure: encoding_dict.pkl, encoding_weights_dict.pkl 

To recreate all plots for the figure: Figure 3.ipynb + Figure 4.ipynb
