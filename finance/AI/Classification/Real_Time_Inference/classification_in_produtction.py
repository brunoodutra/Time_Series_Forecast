#!/usr/bin/env python
# coding: utf-8

# # Real time Performance evaluation of the model

# ## Import libraries

# In[1]:


import time
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# In[2]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# In[3]:


import os, sys
processing_source_path = os.path.abspath('./../../Processing/')
if(processing_source_path not in sys.path):
    sys.path.append(processing_source_path)
from DataLoaderPipeline import FeaturesDataGenerator, scrapingHistoricalData
#from MarketDataCollector import scrapingHistoricalData


# ## Configuring the data parameters and adapters

# In[4]:


SHD=scrapingHistoricalData()

# Lista de criptomoedas
cryptos = ['BTC','ETH']
last_timestamp = pd.Timestamp('2020-02-05 16:00:00')

# Escolha o intervalo
interval = '4h'

# Obtenha os dados históricos
cryptos_df = SHD.get_crypto_historical_data([cryptos[0]], interval, '2024-01-01')


# In[7]:


classifcation_model_path = "./../Experiments/Cryptos/models"
list_of_models =['CNN_MultiHead_2D']

checkpoint_filepath =f'{classifcation_model_path}/model_{list_of_models[0]}_crypto_{cryptos[0]}_best'


# In[8]:


import json
# JSON file
with open(f'{checkpoint_filepath}/config.json', 'r') as file:
    parameters = json.load(file)

for parameter in parameters:
    print(f'{parameter}:{parameters[parameter]}')


# In[9]:


features_indicators=parameters['features_indicators']
features_indicators


# In[10]:


input_shape = (parameters['lookback'], len(features_indicators))

min_norm=parameters['min_norm']
max_norm=parameters['max_norm']
datatype='2D'
trade=['Hold','Buy','Sell']


# In[11]:


dataGen_inference = FeaturesDataGenerator(
    cryptos_df, 
    datatype=datatype, 
    lookback = parameters['lookback'], 
    pred_days = parameters['pred_days'], 
    shuffle= parameters['shuffle'], 
    batch_size=parameters['batch_size'], 
    selected_features = parameters['features_indicators'], 
    data_augmentation=parameters['data_augmentation'], 
    min_max_norm_features=[parameters['min_norm'], parameters['max_norm']]
)


# ### Load the model

# In[12]:


from keras import backend as K
weighted_categorical_crossentropy_loss= dataGen_inference.weighted_categorical_crossentropy(np.ones(3))

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


# In[14]:


import tensorflow_addons as tfa
trained_best_models={}
for model_name in list_of_models:
    print(model_name)
    checkpoint_filepath =f'{classifcation_model_path}/model_{list_of_models[0]}_crypto_{cryptos[0]}_best'
    trained_best_models[f'{model_name}']=tf.keras.models.load_model(
        checkpoint_filepath,
        custom_objects={'loss': weighted_categorical_crossentropy_loss, 'matthews_correlation_coefficient': matthews_correlation_coefficient})


# ### Get the list of all used models

# In[15]:


list_of_models =['CNN_MultiHead_2D']


# In[16]:


parameters['symbol'][0]


# In[17]:


initial_balance = 100


# In[19]:


list_of_trained_models=[]
for crypto in cryptos:
  for model_name in list_of_models:
    checkpoint_filepath =f'{classifcation_model_path}/model_{model_name}_crypto_{crypto}_best'
    with open(f'{checkpoint_filepath}/config.json', 'r') as file:
        parameters = json.load(file)

    trained_model=tf.keras.models.load_model(
            checkpoint_filepath,
            custom_objects={'loss': weighted_categorical_crossentropy_loss, 'matthews_correlation_coefficient': matthews_correlation_coefficient})


    # Escolha o intervalo
    interval = '4h'

    # Obtenha os dados históricos
    cryptos_df = SHD.get_crypto_historical_data([crypto], interval, '2024-01-01')

    dataGen_inference = FeaturesDataGenerator(
        cryptos_df, 
        datatype=datatype, 
        lookback = parameters['lookback'], 
        pred_days = parameters['pred_days'], 
        shuffle= parameters['shuffle'], 
        batch_size=parameters['batch_size'], 
        selected_features = parameters['features_indicators'], 
        data_augmentation=parameters['data_augmentation'], 
        min_max_norm_features=[parameters['min_norm'], parameters['max_norm']]
    )
    
    list_of_trained_models.append({
      "model_name": model_name,
      "configurations": {
          "crypto": parameters['symbol'][0],
          "dataGen_inference":dataGen_inference,
          "trained_model": trained_model,
          "time_operation": interval
      },
      "parameters": {
          "lookback": parameters['lookback'],
          "pred_days": parameters['pred_days'],
          "buy_sell_threshold": [0.05, -0.05],
          "features_indicators": parameters['features_indicators']
        },
      "inference":{
         "TH":parameters['TH'],
         "predition":"Hold",
         "timestamp":last_timestamp,
      },
      "profit":
      {"balance":initial_balance}
      }
    )


# ## Realtime Testing

# In[20]:


# Function to avoid redudant signals 
def generate_signals(signals):
    trade_signals = ['Hold']
    for i in range(1,len(signals)):
        if signals[i] == 'Buy' and signals[i-1] == 'Buy':
            trade_signals.append('Hold')
        elif signals[i] == 'Buy' and signals[i-1] == 'Hold':
            trade_signals.append('Buy')
        elif signals[i] == 'Sell' and signals[i-1] == 'Sell':
            trade_signals.append('Hold')
        elif signals[i] == 'Sell' and signals[i-1] == 'Hold':
            trade_signals.append('Sell')
        else:
            trade_signals.append('Hold')


    return trade_signals


# In[21]:


# Set the interval and data type for the simulation
interval = '4h'
datatype = '2D'  # Adjust the data type
trade = ["Hold", "Buy", "Sell"]  # Define the trade options

# Initialize the simulation parameters
# Initialize the positions list, initial balance, and fees
positions = []
balance = initial_balance
fees = 0.002  # Set the transaction fee


# In[22]:


import pandas as pd
import os

def save_recommendation(model_name=str, crypto=str, data=None, recommendation=None, outuput_percentage=None, timestamp=None):
    # Define the column names
    columns = ['Date', 'Time', 'recommendation', 'percentage','Price', 'Position', 'Quantity']

    # Get the current timestamp
    if timestamp is None:
        import datetime
        timestamp = datetime.datetime.now()

    # Create the result dictionary
    result = {
        'Date': timestamp.date(),
        'Time': timestamp.time(),
        'recommendation': recommendation,
        'percentage' : outuput_percentage,
        'Price': data.loc[len(data)-1, 'Close'],  # Assuming data is a pandas DataFrame
    }

    # Create the file path
    file_path = f'Recommendations/{model_name}_{crypto}_recommendation.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        # Create a new DataFrame and save it to the file
        df = pd.DataFrame([result], columns=columns)
        df.to_csv(file_path, index=False)
    else:
        # Create a new DataFrame and append it to the existing file
        df = pd.DataFrame([result], columns=columns)
        df.to_csv(file_path, mode='a', header=False, index=False)


# In[23]:


def check_and_execute(model=None, dataGen_inference=None, symbol='BTC', TH=[0.5, 0.5, 0.5], last_timestamp=pd.Timestamp('2022-01-01 00:00:00'), interval='4h', balance=100.0):
    """
    Check and execute trades based on the model predictions.
    
    Parameters:
    model (object): The trained model. Default is None.
    dataGen_inference (object): The data generator for inference. Default is None.
    symbol (str): The cryptocurrency symbol. Default is 'BTC'.
    TH (list): The thresholds for buying and selling. Default is [0.5, 0.5, 0.5].
    last_timestamp (pd.Timestamp): The last timestamp. Default is pd.Timestamp('2022-01-01 00:00:00').
    interval (str): The interval for the simulation. Default is '1h'.
    balance (float): The current balance. Default is 100.0.
    
    Returns:
    label_pred (np.array): The predicted labels.
    last_timestamp (pd.Timestamp): The updated last timestamp.
    balance (float): The updated balance.
    """
    
    # Get the current date and time
    today = datetime.today()
    
    # Collect historical data for the last 60 days
    window_days = today - timedelta(days=60)
    start_time = window_days.strftime('%Y-%m-%d')
    data = SHD.get_crypto_historical_data([symbol], interval, start_time)
    current_timestamp = data.loc[len(data)-1,'Date']
    
   
    # Generate features for inference
    x_data_inference = dataGen_inference.comput_features(data, pred_days=0)
    x_data = dataGen_inference.apply_NomrMinmax(x_data_inference, min_norm, max_norm, axis=0)
    
    # Reshape the data for 2D input
    if datatype == '2D':
        x_data = np.transpose(x_data, [0, 2, 1]).reshape(-1, 1, dataGen_inference.inputShape[2], dataGen_inference.inputShape[1])
    
    # Use the trained model and make predictions
    label_pred = model.predict(x_data)
    
    # Generate trade signals based on the predictions
    trade_signals = np.array([
        trade[np.argmax(prediction)] if np.max(prediction) > TH[np.argmax(prediction)] else trade[0]
        for prediction in label_pred
    ])
    
    percentage_signals = np.max(label_pred, axis=1)

    # Generate signals
    trade_signals = generate_signals(trade_signals)
    
    # Print the suggested trades and timestamp
    print(f'Suggested trade of {symbol}: {trade_signals[-2:]} >> Timestamp: {current_timestamp}')
    print("------------------------------------------------------------------------------------")
    #save_recommendation(model_name="CNN", crypto=symbol[0], data=data, recommendation=trade_signals[-2], outuput_percentage=percentage_signals[-2], timestamp=last_timestamp)

    # Check for new trade opportunities
    if last_timestamp != current_timestamp:
        # Check if it's a buy or sell signal
        if trade_signals[-2] != "Hold":
            # Check if it's a buy signal
            if trade_signals[-2] == "Buy":
                # Buy the cryptocurrency
                if balance > 0:  # Only buy if there's a balance
                    price = data.loc[len(data)-1, 'Close']
                    position_value = (balance * (1 - fees)) / price  # Calculate the position value
                    balance = 0  # Zero out the balance
                    positions.append((current_timestamp, price, position_value))
                    print(f'Bought at {current_timestamp} for {price} with {position_value} coins')
            # Check if it's a sell signal
            elif trade_signals[-2] == "Sell":
                # Sell the cryptocurrency
                if positions:  # Only sell if there are positions
                    position = positions.pop(0)
                    price = data.loc[len(data)-1, 'Close']
                    balance = position[2] * price * (1 - fees)  # Calculate the new balance
                    profit = balance - initial_balance
                    position_value = 0  # Zero out the position value
                    print(f'Sold at {current_timestamp} for {price} with balance {balance:.2f} and profit {profit:.2f}')
                else:
                    print(f'No positions to sell at {current_timestamp}')
        
        # save the recommendations
        save_recommendation(model_name="CNN", crypto=symbol, data=data, recommendation=trade_signals[-2], outuput_percentage=percentage_signals[-2], timestamp=last_timestamp)
        # Update the last timestamp and balance
        last_timestamp = current_timestamp
        
    return label_pred, last_timestamp, balance


# In[24]:


list_of_trained_models


# In[ ]:


# Run the simulation in an infinite loop
while True:
    try:
        for idx, trained_model_json in enumerate(list_of_trained_models):
            model_name=trained_model_json['model_name']
            classification_model=trained_model_json['configurations']['trained_model']
            dataGen_inference = trained_model_json['configurations']['dataGen_inference']
            symbol=trained_model_json['configurations']['crypto']
            TH=trained_model_json['inference']['TH']
            last_timestamp=trained_model_json['inference']['timestamp']
            balance=trained_model_json['profit']['balance']

            # Check and execute trades
            label_pred, last_timestamp, balance = check_and_execute(classification_model, dataGen_inference, symbol, TH,last_timestamp, interval, balance)

            list_of_trained_models[idx]['inference']['timestamp'] = last_timestamp
            list_of_trained_models[idx]['profit']['balance'] = balance
            list_of_trained_models[idx]['inference']['predition'] = label_pred

    except Exception as e:
        # Print any exceptions
        print(e)
    # Wait for 20 minutes (1200 seconds) before checking again
    time.sleep(1200)

