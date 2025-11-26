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
import os
from pathlib import Path


# In[2]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# In[ ]:


import os, sys
processing_source_path = os.path.abspath('./../../Processing/')
if(processing_source_path not in sys.path):
    sys.path.append(processing_source_path)
from DataLoaderPipeline import FeaturesDataGenerator, scrapingHistoricalData
#from MarketDataCollector import scrapingHistoricalData

from MarketIndicators import ComputSignalGains

CSG=ComputSignalGains()

# -----------------------------------------------------------------------------
# TFLite Support (Optional)
# -----------------------------------------------------------------------------
# MODEL_BACKEND: 'keras' (default) or 'tflite'. You can override via env var.
MODEL_BACKEND = os.getenv('MODEL_BACKEND', 'keras').lower()
TFLITE_MODEL_PATH = os.getenv('TFLITE_MODEL_PATH', '')
TFLITE_MODEL_DIR = os.getenv('TFLITE_MODEL_DIR', '')

class TFLiteModel:
    """
    Wrapper para executar inferência com modelos TFLite, suportando float32, fp16 e int8.

    - Converte entrada para o dtype esperado pelo modelo.
    - Para int8, aplica quantização usando (x/scale + zero_point).
    - Para saída int8, aplica dequantização usando (y - zero_point) * scale.
    """
    def __init__(self, tflite_path: Path):
        self.tflite_path = Path(tflite_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Handle quantization metadata
        self.input_scale, self.input_zero_point = self.input_details[0].get('quantization', (0.0, 0))
        self.output_scale, self.output_zero_point = self.output_details[0].get('quantization', (0.0, 0))
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']
        self.input_dtype = self.input_details[0]['dtype']
        self.output_dtype = self.output_details[0]['dtype']
        self.input_shape = self.input_details[0]['shape']

    def _ensure_shape(self, x: np.ndarray) -> None:
        """Redimensiona o tensor de entrada quando o batch/shape diverge do esperado."""
        if tuple(x.shape) != tuple(self.input_shape):
            self.interpreter.resize_tensor_input(self.input_index, x.shape)
            self.interpreter.allocate_tensors()
            # Atualiza detalhes após realocação
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_index = self.input_details[0]['index']
            self.output_index = self.output_details[0]['index']
            self.input_dtype = self.input_details[0]['dtype']
            self.output_dtype = self.output_details[0]['dtype']

    def predict_proba(self, x_float: np.ndarray) -> np.ndarray:
        """
        Executa predição em lote e retorna probabilidades em float32.

        Parâmetros:
            x_float: batch de entradas normalizadas em float32.

        Retorna:
            np.ndarray: batch de saídas em float32 (dequantizado quando necessário).
        """
        # Garante batch
        if x_float.ndim == 3:
            # Adiciona canal quando necessário
            x_float = x_float[..., np.newaxis]
        if x_float.ndim == 2:
            x_float = x_float[np.newaxis, ...]

        outputs = []
        for i in range(x_float.shape[0]):
            x_in = x_float[i:i+1]

            # Ajusta shape dinâmico
            self._ensure_shape(x_in)

            # Converte dtype conforme modelo
            if self.input_dtype == np.int8:
                # Para quantização INT8, aplica escala/zero-point
                if self.input_scale == 0:
                    raise ValueError("Input scale 0 detectado para INT8. Verifique o modelo TFLite.")
                x_cast = (x_in / self.input_scale + self.input_zero_point).astype(np.int8)
            else:
                x_cast = x_in.astype(self.input_dtype)

            self.interpreter.set_tensor(self.input_index, x_cast)
            self.interpreter.invoke()
            y = self.interpreter.get_tensor(self.output_index)

            # Dequantiza quando saída é INT8
            if self.output_dtype == np.int8:
                y = (y.astype(np.float32) - self.output_zero_point) * self.output_scale
            else:
                y = y.astype(np.float32)

            outputs.append(y[0])

        return np.stack(outputs, axis=0)

def _resolve_tflite_path() -> Path:
    """
    Resolve o caminho do arquivo TFLite usando variáveis de ambiente.
    Prioridade:
      1) TFLITE_MODEL_PATH
      2) TFLITE_MODEL_DIR contendo model_int8.tflite, model_dynamic.tflite ou model_fp16.tflite
    """
    if TFLITE_MODEL_PATH:
        p = Path(TFLITE_MODEL_PATH)
        if p.exists():
            return p
    if TFLITE_MODEL_DIR:
        base = Path(TFLITE_MODEL_DIR)
        for name in ["model_int8.tflite", "model_dynamic.tflite", "model_fp16.tflite"]:
            cand = base / name
            if cand.exists():
                return cand
        # fallback para qualquer .tflite
        for cand in base.glob("*.tflite"):
            return cand
    raise FileNotFoundError("Arquivo TFLite não encontrado. Defina TFLITE_MODEL_PATH ou TFLITE_MODEL_DIR.")


# ## Configuring the data parameters and adapters

# In[ ]:


SHD=scrapingHistoricalData()

# Lista de criptomoedas
cryptos = ['BTC', 'ETH', 'ADA', 'SOL','XRP']


# Escolha o intervalo
interval = '4h'

# Obtenha os dados históricos
cryptos_df = SHD.get_crypto_historical_data([cryptos[0]], interval, '2024-01-01')

last_timestamp = cryptos_df.loc[len(cryptos_df)-2,'Date']


# In[5]:


classifcation_model_path = "./../Experiments/Cryptos/models"
list_of_models =['CNN_MultiHead_2D']
sufix ='last'
checkpoint_filepath =f'{classifcation_model_path}/model_{list_of_models[0]}_crypto_{cryptos[0]}_{sufix}'


# In[6]:


import json
# JSON file
with open(f'{checkpoint_filepath}/config.json', 'r') as file:
    parameters = json.load(file)

for parameter in parameters:
    print(f'{parameter}:{parameters[parameter]}')


# In[7]:


features_indicators=parameters['features_indicators']
features_indicators


# In[8]:


input_shape = (parameters['lookback'], len(features_indicators))

min_norm=parameters['min_norm']
max_norm=parameters['max_norm']
datatype='2D'
trade=['Hold','Buy','Sell']


# In[9]:


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

# In[10]:


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


# In[11]:


import tensorflow_addons as tfa
trained_best_models={}
for model_name in list_of_models:
    print(model_name)
    checkpoint_filepath =f'{classifcation_model_path}/model_{model_name}_crypto_{cryptos[0]}_{sufix}'
    if MODEL_BACKEND == 'tflite':
        # Usa um único modelo TFLite para todos (geral)
        tflite_path = _resolve_tflite_path()
        trained_best_models[f'{model_name}'] = TFLiteModel(tflite_path)
    else:
        trained_best_models[f'{model_name}']=tf.keras.models.load_model(
            checkpoint_filepath,
            custom_objects={'loss': weighted_categorical_crossentropy_loss, 'matthews_correlation_coefficient': matthews_correlation_coefficient})


# ### Get the list of all used models

# In[12]:


list_of_models =['CNN_MultiHead_2D']


# In[13]:


parameters['symbol'][0]


# In[14]:


initial_balance = 100


# In[15]:


classifcation_model_path


# In[22]:


cryptos


# In[ ]:


list_of_trained_models=[]

for crypto in cryptos:
  for model_name in list_of_models:
    
    checkpoint_filepath =f'{classifcation_model_path}/model_{model_name}_crypto_{crypto}_{sufix}'
    mode= 'unique'

    if os.path.exists(checkpoint_filepath) == False:
       checkpoint_filepath =f'{classifcation_model_path}/model_CNN_restnet_MultiHead_2D_crypto_General_{sufix}'
       mode= 'general'
       
    with open(f'{checkpoint_filepath}/config.json', 'r') as file:
        parameters = json.load(file)
    
    # Seleciona backend do modelo
    if MODEL_BACKEND == 'tflite':
        trained_model = TFLiteModel(_resolve_tflite_path())
        backend = 'tflite'
    else:
        trained_model = tf.keras.models.load_model(
                checkpoint_filepath,
                custom_objects={'loss': weighted_categorical_crossentropy_loss, 'matthews_correlation_coefficient': matthews_correlation_coefficient})
        backend = 'keras'


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
      "mode":mode,
      "configurations": {
          "crypto": crypto,
          "dataGen_inference":dataGen_inference,
          "trained_model": trained_model,
          "backend": backend,
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


# In[31]:


list_of_trained_models[2]


# In[24]:


list_of_trained_models[2]


# ## Realtime Testing

# In[ ]:


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


# In[ ]:


# Set the interval and data type for the simulation
interval = '4h'
datatype = '2D'  # Adjust the data type
trade = ["Hold", "Buy", "Sell"]  # Define the trade options

# Initialize the simulation parameters
# Initialize the positions list, initial balance, and fees
positions = []
balance = initial_balance
fees = 0.002  # Set the transaction fee


# In[ ]:


import pandas as pd
import os

def save_recommendation(model_name=str, crypto=str, data=None, recommendation=None, outuput_percentage=None, timestamp=None, SignalGains=None):
    # Define the column names
    columns = ['Date', 'Time', 'recommendation', 'percentage','Price', 'SignalGains','Position', 'Quantity']

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
    
    # Add SignalGains to result dictionary if it exists
    if SignalGains is not None:
        result['SignalGains'] = json.dumps(SignalGains)


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


# In[ ]:


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
    #TODO change this used time day to a lookback
    window_days = today - timedelta(days=60)
    start_time = window_days.strftime('%Y-%m-%d')
    data_df = SHD.get_crypto_historical_data([symbol], interval, start_time)
    current_timestamp = data_df.loc[len(data_df)-1,'Date']
    
   
    # Generate features for inference
    x_data_inference = dataGen_inference.comput_features(data_df, pred_days=0)
    x_data = dataGen_inference.apply_NomrMinmax(x_data_inference, min_norm, max_norm, axis=0)
    
    # Reshape the data for 2D input
    if datatype == '2D':
        x_data = np.transpose(x_data, [0, 2, 1]).reshape(-1, dataGen_inference.inputShape[1], dataGen_inference.inputShape[2], 1)
        
    # Use the trained model and make predictions
    # Suporta Keras e TFLite
    if hasattr(model, 'predict'):
        label_pred = model.predict(x_data)
    elif isinstance(model, TFLiteModel):
        label_pred = model.predict_proba(x_data)
    else:
        raise ValueError("Modelo não suportado. Esperado Keras Model ou TFLiteModel.")
    
    # Generate trade signalvs based on the predictions
    trade_signals = np.array([
        trade[np.argmax(prediction)] if np.max(prediction) > TH[np.argmax(prediction)] else trade[0]
        for prediction in label_pred
    ])
    
    percentage_signals = np.max(label_pred, axis=1)

    # Generate signals
    #trade_signals = generate_signals(trade_signals)
    
    # Print the suggested trades and timestamp
    print(f'Suggested trade of {symbol}: {trade_signals[-2:]} >> Timestamp: {current_timestamp}')
    print("------------------------------------------------------------------------------------")

    #save_recommendation(model_name="CNN", crypto=symbol[0], data=data, recommendation=trade_signals[-2], outuput_percentage=percentage_signals[-2], timestamp=last_timestamp)

    TradeGains = None
    # Check for new trade opportunities
    if last_timestamp != current_timestamp:
        # Check if it's a buy or sell signal
        if trade_signals[-2] != "Hold":

            # comput the stop gain and targets profit
            #-----------------------------------------------------------
            last_recomendation_window = data_df.iloc[-dataGen_inference.lookback:]

            high_price = last_recomendation_window['Close'].max()
            low_price = last_recomendation_window['Close'].min()
            input_price = last_recomendation_window['Close'].iloc[-1]
            #-----------------------------------------------------------

            # Check if it's a buy signal
            if trade_signals[-2] == "Buy":

                TradeGains = CSG.calculate_Gains_Fibonacci(input_price, low_price, high_price, side=trade_signals[-2])

                # Buy the cryptocurrency
                if balance > 0:  # Only buy if there's a balance
                    price = data_df.loc[len(data_df)-1, 'Close']
                    position_value = (balance * (1 - fees)) / price  # Calculate the position value
                    balance = 0  # Zero out the balance
                    positions.append((current_timestamp, price, position_value))
                    print(f'Bought at {current_timestamp} for {price} with {position_value} coins')
            # Check if it's a sell signal
            elif trade_signals[-2] == "Sell":

                TradeGains = CSG.calculate_Gains_Fibonacci(input_price, high_price, low_price, side=trade_signals[-2])

                # Sell the cryptocurrency
                if positions:  # Only sell if there are positions
                    position = positions.pop(0)
                    price = data_df.loc[len(data_df)-1, 'Close']
                    balance = position[2] * price * (1 - fees)  # Calculate the new balance
                    profit = balance - initial_balance
                    position_value = 0  # Zero out the position value
                    print(f'Sold at {current_timestamp} for {price} with balance {balance:.2f} and profit {profit:.2f}')
                else:
                    print(f'No positions to sell at {current_timestamp}')
        
        # save the recommendations
        save_recommendation(model_name="CNN", crypto=symbol, data=data_df, recommendation=trade_signals[-2], 
                            outuput_percentage=percentage_signals[-2], timestamp=last_timestamp, SignalGains = TradeGains)
        # Update the last timestamp and balance
        last_timestamp = current_timestamp
        
    return label_pred, last_timestamp, balance


# In[ ]:


# Run the simulation in an infinite loop
while True:
    try:
        for idx, trained_model_json in enumerate(list_of_trained_models):
            model_name=trained_model_json['model_name']
            mode=trained_model_json['mode']
            classification_model=trained_model_json['configurations']['trained_model']
            dataGen_inference = trained_model_json['configurations']['dataGen_inference']
            symbol=trained_model_json['configurations']['crypto']
            TH=trained_model_json['inference']['TH']
            last_timestamp=trained_model_json['inference']['timestamp']
            balance=trained_model_json['profit']['balance']
            backend=trained_model_json['configurations'].get('backend', MODEL_BACKEND)

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


# to converto to .py files use this comand:
# 
# jupyter nbconvert --to python .\classification_in_produtction.ipynb
