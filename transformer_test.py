#!/usr/bin/env python
# coding: utf-8


# In[1]:


import xarray as xr
import os
import netCDF4
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from torch.utils.data import random_split
import random
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


# In[2]:


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


# In[3]:


SEQUENCE_SIZE = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
EPOCHS = 1
IN_FEATURES = 124
OUT_FEATURES = 128
# RANDOM_SEED = 42
D_MODEL = 64
# DROPOUT = 0.2
N_HEAD = 4
N_LAYER = 2
MAX_LEN = 5000


# In[4]:


# # use one year
# X=np.load('/work/sds-lab/Shuochen/climsim/val_input.npy')
# y=np.load('/work/sds-lab/Shuochen/climsim/val_target.npy')

# use all 8 years: change path
X_train=np.load('/work/sds-lab/Shuochen/climsim/train_input.npy')
y_train=np.load('/work/sds-lab/Shuochen/climsim/train_target.npy')
X_test=np.load('/work/sds-lab/Shuochen/climsim/val_input.npy')
y_test=np.load('/work/sds-lab/Shuochen/climsim/val_target.npy')

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)


# In[5]:


def create_sliding_window(X, y):
    # to tensor, reshape, transpose (time * space, 124 or 128) > (space, time, 124 or 128)
    X=X.reshape(int(X.shape[0]/384),384,124).permute(1,0,2)
    y=y.reshape(int(y.shape[0]/384),384,128).permute(1,0,2)
    print(X.shape,y.shape)
    # create sliding window (space, time, 124 or 128) > (space, num_window, 124 or 128, seq_size)
    X = X.unfold(1,SEQUENCE_SIZE,1)
    y = y.unfold(1,SEQUENCE_SIZE,1)
    print(X.shape,y.shape)
    # transpose (space, num_window, 124 or 128, seq_size) > (space, num_window, seq_size, 124 or 128)
    X = X.permute(0,1,3,2)
    y = y.permute(0,1,3,2)
    print(X.shape,y.shape)
    # combine dimension (space, num_window, seq_size, 124 or 128) > (space * num_window, seq_size, 124 or 128)
    X = X.reshape(384 * X.shape[1],SEQUENCE_SIZE,124)
    y = y.reshape(384 * y.shape[1],SEQUENCE_SIZE,128)
    print(X.shape,y.shape)
    return X, y

# X, y = create_sliding_window(X, y)
X_train, y_train = create_sliding_window(X_train, y_train)
X_test, y_test = create_sliding_window(X_test, y_test)


# In[6]:


# # # split datasets to training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# print(y_test.shape)

# create datasets
training_set = TensorDataset(X_train, y_train)
testing_set = TensorDataset(X_test, y_test)
# create dataloaders
train_dataloader = DataLoader(training_set, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)
test_dataloader = DataLoader(testing_set,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)


# In[7]:


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=MAX_LEN):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x


# In[8]:


# Model definition using Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim=IN_FEATURES, output_dim = OUT_FEATURES, d_model=D_MODEL, 
                 nhead=N_HEAD, num_layers=N_LAYER):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        # self.relu = nn.ReLU()
        # self.decoder_2 = nn.Linear(512, output_dim)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # x = self.decoder(x[:, -1, :])
        x = self.decoder(x)
        # x = self.relu(x)
        # x = self.decoder_2(x)
        return x

model = TransformerModel().to(device)
sum(p.numel() for p in model.parameters())


# In[9]:


# Train the model
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = EPOCHS
# early_stop_count = 0
# min_val_loss = float('inf')

train_loss_list = []
test_loss_list = []
# this will write training and testing losses after each batch to a txt file. Delete this if not necessary.
f = open("loss_Transformer.txt", "w")
# write hyperparams
f.write(f"SEQUENCE_SIZE={SEQUENCE_SIZE}" + '\t' + f"LEARNING_RATE={LEARNING_RATE}" + '\t' + f"BATCH_SIZE={BATCH_SIZE}" + '\t' + 
            f"EPOCHS={EPOCHS}" + '\t' + f"D_MODEL={D_MODEL}" + '\t' + f"N_HEAD={N_HEAD}" + '\t' + f"N_LAYER={N_LAYER}" + '\t' + f"MAX_LEN={MAX_LEN}" + '\n')

for epoch in range(epochs):
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model.train()
        y_pred = model(X.to(device))
        loss = loss_fn(y_pred, y.to(device))
        train_loss += loss # accumulatively add up the loss per epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    train_loss_list.append(train_loss.detach().cpu().numpy())

    # Validation
    with torch.inference_mode():
        test_loss = 0
        model.eval()
        for X, y in test_dataloader:
            test_pred = model(X.to(device))
            test_loss += loss_fn(test_pred, y.to(device)) # accumulatively add up the loss per epoch
        
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)
        test_loss_list.append(test_loss.detach().cpu().numpy())

    # if test_loss < min_val_loss:
    #     min_val_loss = test_loss
    #     early_stop_count = 0
    # else:
    #     early_stop_count += 1

    # if early_stop_count >= 5:
    #     print("Early stopping!")
    #     break
    
    # print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
    print(f"Epoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

    f.write(str(epoch) + '\t' + f"{float(train_loss):.5f}" + '\t' + f"{float(test_loss):.5f}" + '\n')
    
f.close()


# In[10]:


# plot error curves
df_error = pd.read_csv('loss_Transformer.txt',sep='\t', header=None, skiprows=1)
df_error.columns=['epochs','train_loss','test_loss']
plt.plot(df_error['train_loss'],label='train_loss')
plt.plot(df_error['test_loss'],label='test_loss')
plt.legend()
plt.savefig('Transformer_loss.jpg')


# In[11]:


# combine batches
test_pred = []
with torch.inference_mode():
    test_loss = 0
    model.eval()
    for X, y in test_dataloader:
        test_pred_batch = model(X.to(device))
        test_pred.append(test_pred_batch)

test_pred = torch.cat(tuple(test_pred), dim=0)

def get_original_shape(y):
    # (space * num_window, seq_size, 124 or 128) > (space, num_window, seq_size, 124 or 128)
    y = y.reshape(384, int(y.shape[0]/384),SEQUENCE_SIZE,128)
    # window_1 is values in the first window, take the last value from each rest window and concat to the end.
    # (space, num_window, seq_size, 124 or 128) > (space, time, 124 or 128)
    window_1 = y[:, 0, :, :]
    rest = y[:, 1:, -1, :]
    y = torch.cat((window_1,rest),dim=1)
    y = y.permute(1,0,2)
    y = y.reshape(y.shape[0]*y.shape[1],128)
    return y

# get back the original shape
y_test = get_original_shape(y_test)
test_pred = get_original_shape(test_pred)

# to cpu
test_pred = test_pred.detach().cpu().numpy()


# In[12]:


# plot the first sample. Delete this if not necessary.
plt.plot(test_pred[5000,:])
plt.plot(y_test[5000,:])


# In[ ]:





# # Post-processing (metric files)

# In[13]:


# model name
# (model name is used for the output)
model_name = 'Transformer'
# input of validation dataset (npy)
fn_x_true = '/work/sds-lab/Shuochen/climsim/val_input.npy'
# true output of validation dataset (npy)
fn_y_true = '/work/sds-lab/Shuochen/climsim/val_target.npy'
# Model predicted output of varlidation dataset (npy)
fn_y_pred = test_pred
# model grid information (nc)
fn_grid = '/work/sds-lab/Shuochen/climsim/normalizations_git/ClimSim_low-res_grid-info.nc'

# normalization and scale factors (nc), fn_mli files are not necessary
fn_mli_mean  = '/work/sds-lab/Shuochen/climsim/normalizations_git/inputs/input_mean.nc'
fn_mli_min   = '/work/sds-lab/Shuochen/climsim/normalizations_git/inputs/input_min.nc'
fn_mli_max   = '/work/sds-lab/Shuochen/climsim/normalizations_git/inputs/input_max.nc'
fn_mlo_scale = '/work/sds-lab/Shuochen/climsim/normalizations_git/outputs/output_scale.nc'


# In[14]:


# physical constatns from (E3SM_ROOT/share/util/shr_const_mod.F90)
grav    = 9.80616    # acceleration of gravity ~ m/s^2
cp      = 1.00464e3  # specific heat of dry air   ~ J/kg/K
lv      = 2.501e6    # latent heat of evaporation ~ J/kg
lf      = 3.337e5    # latent heat of fusion      ~ J/kg
ls      = lv + lf    # latent heat of sublimation ~ J/kg
rho_air = 101325./ (6.02214e26*1.38065e-23/28.966) / 273.15 # density of dry air at STP  ~ kg/m^3
                                                            # ~ 1.2923182846924677
                                                            # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
                                                            # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
                                                            # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
rho_h20 = 1.e3       # density of fresh water     ~ kg/m^ 3

vars_mlo_energy_conv = {'ptend_t':cp,
                        'ptend_q0001':lv,
                        'cam_out_NETSW':1.,
                        'cam_out_FLWDS':1.,
                        'cam_out_PRECSC':lv*rho_h20,
                        'cam_out_PRECC':lv*rho_h20,
                        'cam_out_SOLS':1.,
                        'cam_out_SOLL':1.,
                        'cam_out_SOLSD':1.,
                        'cam_out_SOLLD':1.
                       }
vars_longname=\
{'ptend_t':'Heating tendency, ∂T/∂t',
 'ptend_q0001':'Moistening tendency, ∂q/∂t',
 'cam_out_NETSW':'Net surface shortwave flux, NETSW',
 'cam_out_FLWDS':'Downward surface longwave flux, FLWDS',
 'cam_out_PRECSC':'Snow rate, PRECSC',
 'cam_out_PRECC':'Rain rate, PRECC',
 'cam_out_SOLS':'Visible direct solar flux, SOLS',
 'cam_out_SOLL':'Near-IR direct solar flux, SOLL',
 'cam_out_SOLSD':'Visible diffused solar flux, SOLSD',
 'cam_out_SOLLD':'Near-IR diffused solar flux, SOLLD'}


# In[15]:


# set dimemsion names for xarray datasets
dim_name_level  = 'lev'
dim_name_sample = 'sample'

# load input dataset
x_true = np.load(fn_x_true)
y_true = np.load(fn_y_true)
y_pred = fn_y_pred
N_samples = y_pred.shape[0]

# load norm/scale factors
mlo_scale = xr.open_dataset(fn_mlo_scale)
mli_mean  = xr.open_dataset(fn_mli_mean)
mli_min   = xr.open_dataset(fn_mli_min)
mli_max   = xr.open_dataset(fn_mli_max)


# In[16]:


# load grid information
ds_grid = xr.open_dataset(fn_grid) # has ncol:384
N_ncol = len(ds_grid['ncol']) # length of ncol dimension (nlat * nlon)

# make area-weights
ds_grid['area_wgt'] = ds_grid['area'] / ds_grid['area'].mean('ncol')

# map ds_grid's ncol dimension -> the N_samples dimension of npy-loayd arrays (e.g., y_pred)
to_xarray = {'area_wgt': (dim_name_sample,np.tile(ds_grid['area_wgt'], int(N_samples/len(ds_grid['ncol'])))),
            }
to_xarray = xr.Dataset(to_xarray)

# add nsample-mapped grid variables back to ds_grid
ds_grid = xr.merge([ds_grid  [['P0', 'hyai', 'hyam','hybi','hybm','lat','lon','area']],
                    to_xarray[['area_wgt']]])


# In[17]:


# list of ML output variables
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
            'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD'] # mlo mean ML output.

# length of each variable
# (make sure that the order of variables are correct)
vars_mlo_len = {'ptend_t':60,
                'ptend_q0001':60,
                'cam_out_NETSW':1,
                'cam_out_FLWDS':1,
                'cam_out_PRECSC':1,
                'cam_out_PRECC':1,
                'cam_out_SOLS':1,
                'cam_out_SOLL':1,
                'cam_out_SOLSD':1,
                'cam_out_SOLLD':1
               }

# map the length of dimension to the name of dimension
len_to_dim = {60:dim_name_level,
              N_samples: dim_name_sample}


# In[18]:


# Here, we first construct a dictionary of {var name: (dimension name, array-like)},
# then, map the dictionary to an Xarray Dataset.
# (ref: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html)

DS = {}

for kds in ['true', 'pred']:
    if kds=='true':
        work = y_true
    elif kds=='pred':
        work = y_pred

    # [1] Construct dictionary for xarray dataset
    #     format is key for variable name /
    #               value for a turple of (dimension names, data).
    to_xarray = {}
    for k, kvar in enumerate(vars_mlo):

        # length of variable (ie, number of levels)
        kvar_len = vars_mlo_len[kvar]

        # set dimensions of variable
        if kvar_len == 60:
            kvar_dims = (dim_name_sample, dim_name_level)
        elif kvar_len == 1:
            kvar_dims = dim_name_sample

        # set start and end indices of variable in the loaded numpy array
        # then, add 'kvar':(kvar_dims, <np_array>) to dictionary
        if k==0: ind1=0
        ind2 = ind1 + kvar_len

        # scaled output
        kvar_data = np.squeeze(work[:,ind1:ind2])
        # unscaled output
        kvar_data = kvar_data / mlo_scale[kvar].values

        to_xarray[kvar] = (kvar_dims, kvar_data)

        ind1 = ind2

    # [2] convert dict to xarray dataset
    DS[kds] = xr.Dataset(to_xarray)

    # [3] add surface pressure ('state_ps') from ml input
    # normalized ps
    state_ps =  xr.DataArray(x_true[:,120], dims=('sample'), name='state_ps')
    # denormalized ps
    state_ps = state_ps * (mli_max['state_ps'] - mli_min['state_ps']) + mli_mean['state_ps']
    DS[kds]['state_ps'] = state_ps

    # [4] add grid information
    DS[kds] = xr.merge([DS[kds], ds_grid])

    # [5] add pressure thickness of each level, dp
    # FYI, in a hybrid sigma vertical coordinate system, pressure at level z is
    # P[x,z] = hyam[z]*P0 + hybm[z]*PS[x,z],
    # where, hyam and hybm are 
    tmp = DS[kds]['P0']*DS[kds]['hyai'] + DS[kds]['state_ps']*DS[kds]['hybi']
    tmp = tmp.isel(ilev=slice(1,61)).values - tmp.isel(ilev=slice(0,60)).values
    tmp = tmp.transpose()
    DS[kds]['dp'] = xr.DataArray(tmp, dims=('sample', 'lev'))

    # [6] break (sample) to (ncol,time)
    N_timestep = int(N_samples/N_ncol)
    dim_ncol     = np.arange(N_ncol)
    dim_timestep = np.arange(N_timestep)
    new_ind = pd.MultiIndex.from_product([dim_timestep, dim_ncol],
                                         names=['time', 'ncol'])
    DS[kds] = DS[kds].assign_coords(sample=new_ind).unstack('sample')

del work, to_xarray, y_true, y_pred, x_true, state_ps, tmp


# In[19]:


# [1] Weight vertical levels by dp/g that is equivalent to a mass of air within a grid cell per unit area [kg/m2]
# [2] Weight horizontal area of each grid cell by a[x]/mean(a[x]).
# [3] Unit conversion to a common energy unit

DS_ENERGY = {}
for kds in ['true','pred']:
    # Make a copy to keep original dataset
    DS_ENERGY[kds] = DS[kds].copy(deep=True)

    # vertical weighting / area weighting / unit conversion
    for kvar in vars_mlo:

        # [1] weight vertical levels by dp/g
        #     ONLY for vertically-resolved variables, e.g., ptend_{t,q0001}
        # dp/g = - \rho * dz
        if vars_mlo_len[kvar] == 60:
            DS_ENERGY[kds][kvar] = DS_ENERGY[kds][kvar] * DS_ENERGY[kds]['dp']/grav

        # [2] weight area
        #     for ALL variables
        DS_ENERGY[kds][kvar] = DS_ENERGY[kds]['area_wgt'] * DS_ENERGY[kds][kvar]

        # [3] convert units to W/m2
        #     for variables with different units, e.g., ptend_{t,q0001}, precsc, precc
        DS_ENERGY[kds][kvar] =  vars_mlo_energy_conv[kvar] * DS_ENERGY[kds][kvar]


# In[20]:


all_metrics = ['MAE','RMSE','R2']

# A. Calculate metrics
# After this step,
# ptend_{t,q0001} have [ncol, lev] dimension;
# and the rest variables have [ncol] dimension.

# if spatial analysis is desired (e.g., R2 distribution on global map or on latitude-level plane),
# the metrics at this step should be used.


# Select only ML output varibles
DS_ENERGY[kds] = DS_ENERGY[kds][vars_mlo]

# Caclulate 3 metrics
Metrics = {}
Metrics['MAE']  = (np.abs(DS_ENERGY['true']   - DS_ENERGY['pred'])).mean('time')
Metrics['RMSE'] = np.sqrt(((DS_ENERGY['true'] - DS_ENERGY['pred'])**2.).mean('time'))
Metrics['R2'] = 1 - ((DS_ENERGY['true'] - DS_ENERGY['pred']                    )**2.).sum('time')/\
                    ((DS_ENERGY['true'] - DS_ENERGY['true'].mean('time'))**2.).sum('time')

# Save grid-wise metric files in netcdf format
if True:
    for kmetric in ['MAE', 'RMSE', 'R2']:
        fn_save = f'./metrics_netcdf/{model_name}_{kmetric}.nc'
        Metrics[kmetric].to_netcdf(fn_save)


# In[21]:


# B. Make horizontal mean.
# After this step,
# ptend_{t,q0001} have [lev] dimension;
# and the rest variables have zero dimensions, i.e., scalars.

for kmetric in all_metrics:
    Metrics[kmetric] = Metrics[kmetric].mean('ncol') # simple mean


# In[22]:


# C-1. Save the result after B.
# to save in a table format as a csv file, the level dimensions are flattened.

Metrics_stacked = {}
for kmetric in all_metrics:
    Metrics_stacked[kmetric] = Metrics[kmetric].to_stacked_array('ml_out_idx', sample_dims='', name=kmetric)


# save the output
work = pd.DataFrame({'MAE':  Metrics_stacked['MAE'].values,
                     'RMSE': Metrics_stacked['RMSE'].values,
                     'R2':   Metrics_stacked['R2'].values}
                    )
work.index.name = 'output_idx'

# save to the metrics folder
fn_save_metrics = f'./metrics/{model_name}.metrics.csv'
work.to_csv(fn_save_metrics)
work


# In[23]:


# try to reproduce their plot 2
work['MAE'][:60].plot()


# In[24]:


# C-2. Save the result after vertical averaging.
# After this step,
# ptend_{t,q0001} also have zero dimensions, i.e., scalars;

# Then, the results are saved to a csv file.
# This csv file will be used for generating plots.

Metrics_vert_avg = {}
for kmetric in all_metrics:
    Metrics_vert_avg[kmetric] = Metrics[kmetric].mean('lev')
    Metrics_vert_avg[kmetric] = Metrics_vert_avg[kmetric].mean('ilev') # remove dummy dim

# save the output
work = pd.DataFrame({'MAE':  Metrics_vert_avg['MAE'].to_pandas(),
                     'RMSE': Metrics_vert_avg['RMSE'].to_pandas(),
                     'R2':   Metrics_vert_avg['R2'].to_pandas()}
                    )
work.index.name = 'Variable'

# save to the metrics folder
fn_save_metrics_avg = f'./metrics/{model_name}.metrics.lev-avg.csv'
work.to_csv(fn_save_metrics_avg)
work


# In[ ]:




