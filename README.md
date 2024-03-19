# Climsim_test
This repo uses deep learning to improve climate parameterization. The dataset used in this repo is ClimSim: [https://arxiv.org/abs/2306.08754](https://arxiv.org/abs/2306.08754)

## 0. Preprocessing: climsim_data.ipynb
This notebook provides the details about how to get the data from the raw .nc files. This part I only used the last-year data (val_input.npy and val_target.npy).
## 1. Data Downloading
All of the analyses are based on the low-resolution, real-geography dataset. To download the input and output variables for the training and validation sets, go to: [https://huggingface.co/datasets/LEAP/subsampled_low_res/tree/main](https://huggingface.co/datasets/LEAP/subsampled_low_res/tree/main). Download ***train_input.npy***, ***train_target.npy***, ***val_input.npy***, ***val_target.npy***. Or execute ***download_data.ipynb*** to download data from Huggingface directly.
The normalization and scaling files can be found at: [https://github.com/leap-stc/ClimSim/tree/main/preprocessing/normalizations](https://github.com/leap-stc/ClimSim/tree/main/preprocessing/normalizations) These .nc files are required for post processing.
## 2. Baseline model: FCNN.ipynb
Use a one-layer NN to train and test. This will generate metric files for this model. Files will be stored in the ***metrics*** and ***metrics_netcdf*** folders.
## 3. Baseline model: CNN.ipynb
Unfinished.
## 4. Baseline model: Transformer.ipynb
Unfinished.
# Do not use ***quickstart_example.ipynb*** since it does not provide the actual values for MAE/RMSE/R2.