#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 22:28:55 2025

@author: romaindubois
"""

import os
os.chdir("/Users/romaindubois/Library/Mobile Documents/com~apple~CloudDocs/MASTER/MA3/Machine learning/projet/RE_ Mémoire et machine learning")


################ LSTM ############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

###### loading data ######################################

df_precipitation_raw = pd.read_excel("RhiresDD_61_23_Gorner.xlsx")
df_temp_raw = pd.read_excel("TabsD_65_23_Gorner.xlsx")
df_flow_raw = pd.read_excel("Gornera_disch_corr.xlsx")

###### pre-porcessing data ##############################


#precipitation et temp

# Passer de "wide" à "long"
df_long_p = df_precipitation_raw.melt(id_vars="Day/Year", 
                      var_name="year", 
                      value_name="precip_mm")

df_long_T = df_temp_raw.melt(id_vars="Day/Year", 
                      var_name="year", 
                      value_name="temp_C")

# Convertir les colonnes en numérique
df_long_p["year"] = df_long_p["year"].astype(int)
df_long_T["year"] = df_long_T["year"].astype(int)

# Créer une vraie date à partir de year + day of year
df_long_p["date"] = pd.to_datetime(df_long_p["year"].astype(str), format="%Y") \
                  + pd.to_timedelta(df_long_p["Day/Year"] - 1, unit="D")
                  
# Créer une vraie date à partir de year + day of year
df_long_T["date"] = pd.to_datetime(df_long_T["year"].astype(str), format="%Y") \
                  + pd.to_timedelta(df_long_T["Day/Year"] - 1, unit="D")

# Mettre la date en index
df_precip = df_long_p[["date", "precip_mm"]].set_index("date")
df_temp = df_long_T[["date", "temp_C"]].set_index("date")

# -------------------------
# débit (15-min → journalier moyen)
# -------------------------

# Passer du format "wide" à "long"
df_flow_long = df_flow_raw.melt(
    id_vars="Time_Matlab",
    var_name="year",
    value_name="debit_m3s"
)

df_flow_long["year"] = df_flow_long["year"].astype(int)

# Construire une vraie date : année + fraction de jour MATLAB
df_flow_long["date"] = pd.to_datetime(df_flow_long["year"].astype(str), format="%Y") \
                      + pd.to_timedelta(df_flow_long["Time_Matlab"], unit="D")

# Série temporelle complète à 15 min
df_debit_15min = df_flow_long[["date", "debit_m3s"]].set_index("date").sort_index()

# 👉 Convertir le débit 15 min en débit journalier moyen
df_debit_daily = df_debit_15min.resample("D").mean()

df_debit_daily.columns = ["debit_mean"]  # nom clair pour la suite

print("Shape débit journalier :", df_debit_daily.shape)
print(df_debit_daily.head())

#surface area glacier


# Années et surfaces connues (km²)
years = np.array([1973, 2010, 2016])
areas = np.array([57.77, 40.24, 41.23])
# -----------------------------
# 2. Régression linéaire
# -----------------------------
coeffs = np.polyfit(years, areas, 1)
slope, intercept = coeffs

print("Slope:", slope)
print("Intercept:", intercept)

# -----------------------------
# 3. Interpolation : années 1971 → 2023
# -----------------------------
years_full = np.arange(1971, 2024)
areas_interp = slope * years_full + intercept
surface_area = pd.DataFrame({
    "year": years_full,
    "surface_km2": areas_interp
})

###### FUSION DES 4 TYPES DE DONNÉES #########################

# 1) Surface du glacier → convertir en dates journalières

# Convertir year → timestamp (1er janvier)
surface_area["date"] = pd.to_datetime(surface_area["year"].astype(str) + "-01-01")
surface_area = surface_area.set_index("date").sort_index()

# Étaler la surface glaciaire par forward-fill sur l’année
df_glacier_daily = surface_area[["surface_km2"]].resample("D").ffill()

print("Glacier daily :", df_glacier_daily.shape)

# -------------------------------------------------------

# 2) Préparer précipitation et température
df_precip.columns = ["precip_mm"]
df_temp.columns = ["temp_C"]

print("Precip shape :", df_precip.shape)
print("Temp shape   :", df_temp.shape)

# -------------------------------------------------------

# 3) Débit déjà transformé en moyenne journalière
df_debit_daily.columns = ["debit_mean"]

print("Debit daily :", df_debit_daily.shape)

# -------------------------------------------------------

# 4) Fusion complète (inner join sur les dates communes)
df_all = (
    df_precip
    .join(df_temp, how="inner")
    .join(df_glacier_daily, how="inner")
    .join(df_debit_daily, how="inner")
)

df_all = df_all.sort_index()

# Supprimer les lignes où la cible 'debit_mean' est NaN
df_all_clean = df_all.dropna(subset=['debit_mean'])

# Vérifier qu'il n'y a plus de NaN dans la cible
df_all_clean['debit_mean'].isna().sum()

# Extract day of year from datetime index
doy = df_all_clean.index.dayofyear

# Handle leap years by normalizing with 365.25
df_all_clean["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df_all_clean["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

# la colonne débit redevient la dernière colonne

df_all_clean = df_all_clean[list(df_all_clean.columns[:3]) + list(df_all_clean.columns[4:]) + [df_all_clean.columns[3]]]


########################## split train-valid-test##########################

years_available = df_all_clean.index.year.unique()  # Find the unique year values
print(len(years_available))

test_years = years_available[-10:]
print(test_years)

# Remove the last 10 years (reserved for the test set)
remaining_years = years_available[:-10]

# 10 last years also
validation_years = remaining_years[-10:]
print(validation_years)

test_set = df_all_clean[df_all_clean.index.year.isin(test_years)]
valid_set = df_all_clean.loc[df_all_clean.index.year.isin(validation_years)]
train_set = df_all_clean.loc[~df_all_clean.index.year.isin([*test_years, *validation_years])]

print(train_set.head(), valid_set.head(), test_set.head(), sep='\n\n')

############# scalling ################

scaler = ColumnTransformer(remainder='passthrough',transformers=[('scaling', StandardScaler(),['precip_mm', 'temp_C', 'surface_km2','doy_sin','doy_cos'])])


train = scaler.fit_transform(train_set)   # fit the scaler and transform the train_set
val = scaler.transform(valid_set)           # transform the val_set
test = scaler.transform(test_set)         # transform the test_set

# Make sure your datasets have been scaled correctly
print(train[:2], val[:2],test[:2], sep='\n\n')


################## GRID SEARCH WINDOW SIZE ##########################

window_sizes_to_test = [1,2,3,5,7, 14, 30, 90, 180, 365]
window_results = {}   # pour stocker NSE de validation
window_results2 = {}   # pour stocker KGE de validation

for window_size in window_sizes_to_test:
    print(f"\n\n===== TESTING WINDOW SIZE = {window_size} =====")
    sequence_length = window_size  # pour garder ta variable

    ################## on créé les winodws ##########################

    train_X = np.lib.stride_tricks.sliding_window_view(train[:,:5],  # source array
                                                       window_size,   # window size
                                                       axis=0)        # Window Sliding Direction

    val_X = np.lib.stride_tricks.sliding_window_view(val[:,:5],      # source array
                                                     window_size,
                                                     axis=0)

    test_X = np.lib.stride_tricks.sliding_window_view(test[:,:5],    # source array
                                                      window_size,
                                                      axis=0)

    train_X = np.moveaxis(train_X, 1, 2)
    val_X = np.moveaxis(val_X, 1, 2)
    test_X = np.moveaxis(test_X, 1, 2)

    train_y = train[window_size-1:, 5]
    val_y   = val[window_size-1:, 5]
    test_y  = test[window_size-1:, 5]


    calc_device = torch.device('cpu')


    ############### on convertit les données en tensors ############################

    train_Xtensor = torch.from_numpy(train_X).float().to(calc_device)
    train_ytensor = torch.from_numpy(train_y).float().to(calc_device)

    val_Xtensor = torch.from_numpy(val_X).float().to(calc_device)
    val_ytensor = torch.from_numpy(val_y).float().to(calc_device)

    test_Xtensor = torch.from_numpy(test_X).float().to(calc_device)
    test_ytensor = torch.from_numpy(test_y).float().to(calc_device)

    train_data = TensorDataset(train_Xtensor, train_ytensor)
    val_data   = TensorDataset(val_Xtensor,   val_ytensor)
    test_data  = TensorDataset(test_Xtensor,  test_ytensor)

    #Define the hyperparameters
    batch_size = 128
    num_workers = 0

   ############## data loaders ###################################

    train_loader = DataLoader(
       dataset=train_data,    # Define the dataset to use
       batch_size=batch_size,         # Set the batch size
       shuffle=True,num_workers=num_workers          # Shuffle for training
    )

    val_loader = DataLoader(
       dataset=val_data,      # Define the dataset to use
       batch_size=batch_size,         # Set the batch size
       shuffle=False,num_workers=num_workers          # No shuffle for validation
    )

    test_loader = DataLoader(
       dataset=test_data,     # Define the dataset to use
       batch_size=batch_size,         # Set the batch size
       shuffle=False,num_workers=num_workers          # No shuffle for testing
    )


    #################### LSTM architecture (identique) ############################

    ############## LSTM class ########################################

    class MyLSTM(nn.Module):  # Define the class which we're extending
        # Initialization
        def __init__(self,
                     hidden_size,   # Model Hyperparameter 1
                     dropout_rate): # Model Hyperparameter 2

            super(MyLSTM, self).__init__()

            # Store hyperparameters
            self.hidden_size = hidden_size
            self.dropout_rate = dropout_rate

            # Define the LSTM layer
            self.LSTM_layer = nn.LSTM(
                input_size = 5,               # The number of input features (T, P1, P2)
                hidden_size = self.hidden_size,
                num_layers = 1,               # One LSTM layer
                bias = True,                  # Enable biases
                batch_first = True            # (batch, seq, features)
            )

            # Dropout layer
            self.dropout_layer = nn.Dropout(
                p = self.dropout_rate
            )

            # Output layer
            self.out_layer = nn.Linear(
                in_features = self.hidden_size,
                out_features = 1               # One prediction per sequence
            )

        # Forward pass
        def forward(self, X):

            # LSTM forward pass
            output, (h_n, c_n) = self.LSTM_layer(X)

            # h_n has shape (num_layers, batch_size, hidden_size)
            hidden_state = self.dropout_layer(h_n[0])  # Take last layer’s hidden state

            # Prediction
            p_hat = torch.flatten(self.out_layer(hidden_state))


            return p_hat

    ################ NSE as performance metric #######################

    def calc_nse(sim: torch.FloatTensor, obs: torch.FloatTensor, global_obs_mean: torch.FloatTensor) -> float:
        """Calculate the Nash-Sutcliff-Efficiency coefficient.

        :param obs: Array containing the observations
        :param sim: Array containing the simulations
        :param global_obs_mean: mean of the whole observation series
        :return: NSE value.
        """
        numerator = torch.square(sim - obs).sum()
        denominator = torch.square(obs - global_obs_mean).sum()
        nse_val = 1 - numerator / denominator

        return nse_val

    ############## NSE as performance metric ##########################

    def calc_kge(sim: torch.FloatTensor, obs: torch.FloatTensor) -> float:
        """Calculate the Kling-Gupta Efficiency (KGE).

        :param sim: Tensor containing the simulations
        :param obs: Tensor containing the observations
        :return: KGE value
        """

        # Moyennes
        mean_sim = torch.mean(sim)
        mean_obs = torch.mean(obs)

        # Écarts-types
        std_sim = torch.std(sim, unbiased=False)
        std_obs = torch.std(obs, unbiased=False)

        # Corrélation de Pearson
        r = torch.sum((sim - mean_sim) * (obs - mean_obs)) / (
            torch.sqrt(torch.sum((sim - mean_sim) ** 2)) *
            torch.sqrt(torch.sum((obs - mean_obs) ** 2))
        )

        # Termes alpha et beta
        alpha = std_sim / std_obs
        beta = mean_sim / mean_obs

        # KGE
        kge_val = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        return kge_val

    # Define the loss function for training
    loss_func = nn.MSELoss()

    # Weighted MSE to penalize errors more when discharge is high
    """def weighted_mse(pred, obs, alpha=2.0):
        
        Weighted MSE: weights increase with observed discharge.
        alpha controls how much weight high flows receive.
        
        weights = 1 + alpha * (obs / obs.mean())
        return torch.mean(weights * (pred - obs)**2)"""

    # Define the loss function for training (weighted MSE)
    #loss_func = lambda pred, obs: weighted_mse(pred, obs, alpha=2.0)"""

    # Instantiate our LSTM model
    model = MyLSTM(hidden_size = 16,     # Hyperparameter 1
                   dropout_rate = 0.125  # Hyperparameter 2
                  ).to(calc_device)      # Send to CPU/GPU

    # Define our optimizer
    optimizer = torch.optim.Adam(model.parameters(),   # Parameters to optimize
                                 lr = 1e-3)            # Learning rate

    # Define the number of epochs
    num_epochs = 50

    def eval_model(model,          # the model to be evaluated
                   dataloader,     # the dataloader for the dataset used for the evaluation
                   loss_func,      # the main loss function to be used
                   metric_func):   # the function to be used as a performance metric

        with torch.no_grad():

            # Zero the loss and the metric
            loss = 0
            metric = 0

            # Compute dataset-wide mean (needed for NSE)
            global_sum = 0
            label_size = 0

            # First pass: compute mean of observations
            for X_batch, y_batch in dataloader:
                global_sum += y_batch.sum()
                label_size += len(y_batch)

            global_mean = global_sum / label_size

            # Second pass: compute predictions, loss, metric
            for X_batch, y_batch in dataloader:
                preds = model(X_batch)

                # batch loss
                batch_loss = loss_func(preds, y_batch)

                # batch metric (e.g. NSE)
                batch_metric = metric_func(preds, y_batch, global_mean)
                

                loss += batch_loss.item()
                metric += batch_metric.item()

            # Number of batches
            num_batches = len(dataloader)

            # Averages
            loss = loss / num_batches
            metric = metric / num_batches

            return (loss, metric)

    ##################### Training loop (identique) #########################
    train_losses = []
    val_losses = []
    val_NSEs = []
 
    
    for epoch in range(num_epochs):
         # Zero the training loss
         train_loss = 0
    
         # Iterate through the features and labels in the train dataloader
         for X_batch, y_batch in train_loader:
    
             # Zero gradients
             optimizer.zero_grad()
    
             # Predictions
             pred = model(X_batch)
    
             # Batch loss
             batch_loss = loss_func(pred, y_batch)
    
             # Backpropagation
             batch_loss.backward()
    
             # Optimizer update
             optimizer.step()
    
             # Accumulate training loss
             train_loss += batch_loss.item()
    
         # Number of batches
         num_batches = len(train_loader)
    
         # Mean training loss
         train_loss = train_loss / num_batches
    
         # Store training loss
         train_losses.append(train_loss)
    
         # Validation evaluation
         val_loss, val_NSE = eval_model(model,
                                        val_loader,
                                        loss_func,
                                        calc_nse)
         
    
         # Store validation metrics
         val_losses.append(val_loss)
         val_NSEs.append(val_NSE)
         
        
    
         # Save best model
         if val_NSE >= max(val_NSEs):
             #torch.save(model, './best_model.pt')
             torch.save(model.state_dict(), './best_model.pt')
    
    
         # Logging
         print(f'\rEpoch: {epoch+1}/{num_epochs}, '
               f'train_loss: {train_loss}, '
               f'val_loss: {val_loss}, '
               f'NSE: {val_NSE}',
               
               end="")

    # stocker la meilleure NSE pour cette fenêtre
    window_results[window_size] = max(val_NSEs)
    
    

    # Affichage final
    print("\n\n======= VALIDATION NSE BY WINDOW SIZE =======")
    for w, nse, in window_results.items():
        print(f"Window {w} → NSE = {nse:.4f}")
    
       
    
    best_window = max(window_results, key=window_results.get)
    print(f"\nBEST WINDOW SIZE = {best_window} (NSE = {window_results[best_window]:.4f})")


############# utilisation de la meilleure fenetre temporelle sur le test set

best_window = 365  # obtenu grâce ton grid search
window_size = best_window

# Recreate windows for the test set using the BEST window
train_X = np.lib.stride_tricks.sliding_window_view(train[:,:3], window_size, axis=0)
val_X   = np.lib.stride_tricks.sliding_window_view(val[:,:3],   window_size, axis=0)
test_X  = np.lib.stride_tricks.sliding_window_view(test[:,:3],  window_size, axis=0)

train_X = np.moveaxis(train_X, 1, 2)
val_X   = np.moveaxis(val_X, 1, 2)
test_X  = np.moveaxis(test_X, 1, 2)

train_y = train[window_size-1:, 3]
val_y   = val[window_size-1:, 3]
test_y  = test[window_size-1:, 3]

print("Recreated windows for BEST window size:", best_window)
print("Test X shape:", test_X.shape)

calc_device = torch.device('cpu')

test_Xtensor = torch.from_numpy(test_X).float().to(calc_device)
test_ytensor = torch.from_numpy(test_y).float().to(calc_device)

test_data  = TensorDataset(test_Xtensor,  test_ytensor)
test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)

model = MyLSTM(hidden_size=16, dropout_rate=0.125).to(calc_device)
model.load_state_dict(torch.load(f'best_model_window_{best_window}.pt'))
model.eval()
print("Loaded model from best window size:", best_window)

with torch.no_grad():
    preds_test = model(test_Xtensor).cpu().numpy()
    obs_test = test_ytensor.cpu().numpy()

mse_test, nse_test = eval_model(model, test_loader, loss_func, calc_nse)
print(f"Test MSE = {mse_test:.4f}")
print(f"Test NSE = {nse_test:.4f}")


with torch.no_grad():
    fig, ax = plt.subplots(figsize=(20,5), dpi=150)

    ax.plot(test_set.index[window_size-1:],
            obs_test,
            c='teal',
            linewidth=1,
            label='Observed')

    ax.plot(test_set.index[window_size-1:],
            preds_test,
            c='orange',
            alpha=0.8,
            linewidth=1,
            label='Predicted')

    ax.axhline(np.mean(obs_test), color='teal', alpha=0.5, linestyle='--', label='Obs Mean')

    ax.legend()
    ax.set_title(f"Predictions on Test Set (window={best_window})\nMSE={mse_test:.2f}, NSE={nse_test:.2f}")
    ax.set_ylabel("Discharge (m³/s)")

    ax.autoscale(enable=True, axis='x', tight=True)
    fig.set_facecolor('lightgrey')
    plt.show()

