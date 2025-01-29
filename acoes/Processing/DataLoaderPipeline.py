from tqdm import tqdm
import time
from keras.utils import Sequence
from keras import backend as K
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import resample
import requests
from imblearn.over_sampling import RandomOverSampler 

import sys

class ComputIndicators():
    
    def __init__(self):
        self._lambda = 22e-12

    def autorregressive_coefs(self, data, p=3):
        channels = data.shape[0]
        phi = np.zeros([data.shape[0], p])
        ar_coefs = np.zeros([channels, p])
        y_init = []
        y = data
        aux = np.zeros(p)
        for k in range(p): 
            aux[p-k:] = y[k]
            y_init.append(aux.copy())
        phi = np.vstack([y[i-p:i] if i-p>=0 else y_init[i]  for i in range(0, len(y))])
        ar_coefs[:] = np.linalg.inv(phi.T.dot(phi)).dot(phi.T.dot(data))
        return ar_coefs

    def moving_average(self, data, window_length):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        moving_average_values = np.zeros(len(data))
        for i in range(len(data)):
            if i < window_length - 1:
                moving_average_values[i] = np.mean(data[:i+1])
            else:
                window_slice = data[i - window_length + 1 : i + 1]
                moving_average_values[i] = np.mean(window_slice)
        return moving_average_values

    def exponential_moving_average(self, data, window_length):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        ema = np.zeros(len(data))
        ema[:window_length] = np.mean(data[:window_length])
        alpha = 2 / (1 + window_length)  
        for i in range(window_length, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        if any(period < 1 for period in [fast_period, slow_period, signal_period]):
            raise ValueError("Window lengths must be positive integers.")
        ema_fast = self.exponential_moving_average(data, fast_period)
        ema_slow = self.exponential_moving_average(data, slow_period)
        macd = ema_fast - ema_slow
        macd_signal = self.exponential_moving_average(macd, signal_period)
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def SCP(self, close_prices, window_length=12):
        #stationary_closing_price
        if len(close_prices) < window_length:
            raise ValueError("The length of close prices must be greater than the window length.")
        scp = np.zeros(len(close_prices))
        for i in range(len(close_prices)):
            if i < window_length:
                scp[i] = 0
            else:
                scp[i] = np.tanh(close_prices[i] - close_prices[i-1])
        return scp
    
    def rsi(self, data, period=14, pred_days = 1):
        if period < 1:
            raise ValueError("Period must be a positive integer.")
        delta = np.diff(data)
        up_changes = np.where(delta > 0, delta, 0)
        down_changes = np.where(delta < 0, np.abs(delta), 0)
        avg_gain = np.zeros(len(data))
        avg_loss = np.zeros(len(data))
        avg_gain[:period] = np.cumsum(up_changes[:period]) / np.arange(1, period + 1)
        avg_loss[:period] = np.cumsum(down_changes[:period]) / np.arange(1, period + 1)
        for i in range(period, len(up_changes)):
            avg_gain[i + 1] = (avg_gain[i] * (period - 1) + up_changes[i]) / period
            avg_loss[i + 1] = (avg_loss[i] * (period - 1) + down_changes[i]) / period
        epsilon = 1e-8
        rs = avg_gain / (avg_loss + epsilon)
        rs = np.where(avg_loss == 0, np.inf, rs)
        rsi = 100 - (100 / (1 + rs))
        return rsi
      
    def cci(self, high_prices, low_prices, close_prices, window_length=20):
        typical_prices = (high_prices + low_prices + close_prices) / 3
        sma_typical_prices = self.moving_average(typical_prices, window_length)
        mean_deviation = np.mean(np.abs(typical_prices - sma_typical_prices))
        cci_values = (typical_prices - sma_typical_prices) / (0.015 * mean_deviation + self._lambda)
        return cci_values

    def stochastic(self, high_prices, low_prices, close_prices, window_length=14, smooth_k=3, smooth_d=3):
        if window_length < 1 or smooth_k < 1 or smooth_d < 1:
            raise ValueError("Window lengths must be positive integers.")
        lowest_low = self.minimum(low_prices, window_length)
        highest_high = self.maximum(high_prices, window_length)
        percent_k = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
        percent_d = self.moving_average(percent_k, window_length=smooth_k)
        return percent_k, percent_d

    def bollinger_bands(self, data, window_length=20, num_std=2):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        moving_average = self.moving_average(data, window_length)
        std_deviation = np.zeros(len(data))
        for i in range(len(data)):
            if i < window_length - 1:
                std_deviation[i] = np.std(data[:i+1])
            else:
                std_deviation[i] = np.std(data[i - window_length + 1: i + 1])
        upper_band = moving_average + num_std * std_deviation
        lower_band = moving_average - num_std * std_deviation
        return moving_average, upper_band, lower_band

    def minimum(self, data, window_length):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        minimum_values = np.zeros(len(data))
        for i in range(len(data)):
            if i < window_length - 1:
                minimum_values[i] = np.min(data[:i+1])
            else:
                window_slice = data[i - window_length + 1: i + 1]
                minimum_values[i] = np.min(window_slice)
        return minimum_values

    def maximum(self, data, window_length):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        maximum_values = np.zeros(len(data))
        for i in range(len(data)):
            if i < window_length - 1:
                maximum_values[i] = np.max(data[:i+1])
            else:
                window_slice = data[i - window_length + 1: i + 1]
                maximum_values[i] = np.max(window_slice)
        return maximum_values

    def momentum(self, data, window_length=10):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        momentum_values = np.zeros(len(data))
        for i in range(len(data)):
            if i < window_length - 1:
                momentum_values[i] = data[i]
            else:
                momentum_values[i] = data[i] - data[i - window_length]
        return momentum_values

    def roc(self, data, window_length=10):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        roc_values = np.zeros(len(data))
        for i in range(len(data)):
            if i < window_length - 1:
                roc_values[i] = 0
            else:
                roc_values[i] = (data[i] - data[i - window_length]) / data[i - window_length] * 100
        return roc_values

    def on_balance_volume(self, close_prices, volumes):
        if len(close_prices) != len(volumes):
            raise ValueError("The lengths of close prices and volumes must be equal.")
        obv = np.zeros(len(close_prices))
        obv[0] = volumes[0]
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif close_prices[i] < close_prices[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]
        return obv

    def accumulation_distribution_line(self, high_prices, low_prices, close_prices, volumes):
        if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices) or len(high_prices) != len(volumes):
            raise ValueError("The lengths of all input arrays must be equal.")
        money_flow = np.zeros(len(high_prices))
        for i in range(len(high_prices)):
            money_flow[i] = ((high_prices[i] + low_prices[i] + close_prices[i]) / 3) * volumes[i]
        ad_line = np.zeros(len(high_prices))
        ad_line[0] = money_flow[0]
        for i in range(1, len(high_prices)):
            ad_line[i] = ad_line[i - 1] + money_flow[i]
        return ad_line

    def money_flow_index(self, high_prices, low_prices, close_prices, volumes, window_length=14):
        """
        Calculate the Money Flow Index (MFI).
        
        Parameters:
            high_prices (array-like): Array of high prices.
            low_prices (array-like): Array of low prices.
            close_prices (array-like): Array of close prices.
            volumes (array-like): Array of volumes.
            window_length (int): Number of periods for calculation (default: 14).
        
        Returns:
            np.ndarray: MFI values with NaN for periods without enough data.
        """
        if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices) or len(high_prices) != len(volumes):
            raise ValueError("All input arrays must have the same length.")
        if len(high_prices) < window_length:
            raise ValueError("Input data must have at least 'window_length' elements.")
        # Calculate Typical Price (TP)
        typical_price = (high_prices + low_prices + close_prices) / 3

        # Calculate Money Flow (MF)
        money_flow = typical_price * volumes

        # Determine Positive and Negative Money Flows
        positive_flow = np.where(typical_price[1:] > typical_price[:-1], money_flow[1:], 0)
        negative_flow = np.where(typical_price[1:] < typical_price[:-1], money_flow[1:], 0)

        # Initialize MFI array
        mfi = np.full(len(typical_price), self._lambda)

        # Calculate MFI using rolling sums
        for i in range(window_length - 1, len(typical_price)):
            positive_sum = np.sum(positive_flow[i - window_length + 1:i])
            negative_sum = np.sum(negative_flow[i - window_length + 1:i])
            
            if negative_sum == 0:
                mfi[i] = 100
            else:
                money_flow_ratio = positive_sum / (negative_sum)
                mfi[i] = 100 - (100 / (1 + money_flow_ratio))

        return mfi

    def ichimoku_cloud(self, high_prices, low_prices, window_length1=9, window_length2=26, window_length3=52):
        if window_length1 < 1 or window_length2 < 1 or window_length3 < 1:
            raise ValueError("Window lengths must be positive integers.")
        Tenkan_sen = (self.maximum(high_prices, window_length1) + self.minimum(low_prices, window_length1)) / 2
        Kijun_sen = (self.maximum(high_prices, window_length2) + self.minimum(low_prices, window_length2)) / 2
        Senkou_span_a = (Tenkan_sen + Kijun_sen) / 2
        Senkou_span_b = (self.maximum(high_prices, window_length3) + self.minimum(low_prices, window_length3)) / 2
        return Tenkan_sen, Kijun_sen, Senkou_span_a, Senkou_span_b

    def parabolic_sar(self, high_prices, low_prices, acceleration=0.02, maximum=0.2):
        if acceleration < 0 or maximum < 0:
            raise ValueError("Acceleration and maximum must be non-negative.")
        sar = np.zeros(len(high_prices))
        sar[0] = low_prices[0]
        direction = 1
        for i in range(1, len(high_prices)):
            if direction == 1:
                sar[i] = sar[i - 1] + acceleration * (high_prices[i - 1] - sar[i - 1])
                if low_prices[i] < sar[i]:
                    direction = -1
                    sar[i] = low_prices[i]
            else:
                sar[i] = sar[i - 1] - acceleration * (low_prices[i - 1] - sar[i - 1])
                if high_prices[i] > sar[i]:
                    direction = 1
                    sar[i] = high_prices[i]
            if acceleration > maximum:
                acceleration = maximum
        return sar

    def average_directional_index(self, high_prices, low_prices, close_prices, window_length=14):
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        plus_di = np.zeros(len(high_prices))
        minus_di = np.zeros(len(high_prices))
        for i in range(1, len(high_prices)):
            plus_dm = high_prices[i] - high_prices[i - 1]
            minus_dm = low_prices[i - 1] - low_prices[i]
            if plus_dm > minus_dm and plus_dm > 0:
                plus_di[i] = plus_dm
            if minus_dm > plus_dm and minus_dm > 0:
                minus_di[i] = minus_dm
        plus_di = self.moving_average(plus_di, window_length)
        minus_di = self.moving_average(minus_di, window_length)
        adx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        return adx

    def fibonacci_retracements(self, high_price, low_price):
        if high_price < low_price:
            raise ValueError("High price must be greater than or equal to low price.")
        levels = [low_price, low_price + (high_price - low_price) * 0.236, low_price + (high_price - low_price) * 0.382, 
                  low_price + (high_price - low_price) * 0.5, low_price + (high_price - low_price) * 0.618, low_price + (high_price - low_price) * 0.764, high_price]
        return levels

    def candlestick_patterns(self, open_prices, high_prices, low_prices, close_prices):
        if len(open_prices) != len(high_prices) or len(open_prices) != len(low_prices) or len(open_prices) != len(close_prices):
            raise ValueError("The lengths of all input arrays must be equal.")
        patterns = []
        for i in range(len(open_prices)):
            if close_prices[i] > open_prices[i] and high_prices[i] > close_prices[i] and low_prices[i] < open_prices[i]:
                patterns.append(" Hammer")
            elif close_prices[i] < open_prices[i] and high_prices[i] > open_prices[i] and low_prices[i] < close_prices[i]:
                patterns.append("Shooting Star")
            # Add more patterns as needed
        return patterns

    def elliott_wave_theory(self, high_prices, low_prices):
        if len(high_prices) != len(low_prices):
            raise ValueError("The lengths of high prices and low prices must be equal.")
        waves = []
        for i in range(len(high_prices)):
            if high_prices[i] > high_prices[i - 1] and low_prices[i] > low_prices[i - 1]:
                waves.append("Impulse Wave")
            elif high_prices[i] < high_prices[i - 1] and low_prices[i] < low_prices[i - 1]:
                waves.append("Corrective Wave")
            # Add more wave patterns as needed
        return waves
    
    def chaikin_money_flow(self, high_prices, low_prices, close_prices, volumes, window_length=21):
        if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices) or len(high_prices) != len(volumes):
            raise ValueError("The lengths of all input arrays must be equal.")
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        
        multipliers = np.zeros(len(high_prices))
        money_flow_volumes = np.zeros(len(high_prices))
        for i in range(len(high_prices)):
            high = high_prices[i]
            low = low_prices[i]
            close = close_prices[i]
            volume = volumes[i]
            
            multiplier = ((close - low) - (high - close)) / (high - low + sys.float_info.epsilon)
            money_flow_volume = volume * multiplier 
            
            multipliers[i] = multiplier
            money_flow_volumes[i] = money_flow_volume
        
        cmf = np.zeros(len(high_prices))
        for i in range(window_length, len(high_prices)):
            window_slice = money_flow_volumes[i - window_length + 1: i + 1]
            volume_window_slice = volumes[i - window_length + 1: i + 1]
            cmf[i] = np.sum(window_slice) / (np.sum(volume_window_slice) +sys.float_info.epsilon)
    
        return cmf
    
    def rate_of_change(self, close_prices, window_length=14):
        roc = np.zeros(len(close_prices))
        for i in range(window_length, len(close_prices)):
            roc[i] = ((close_prices[i] - close_prices[i - window_length]) / close_prices[i - window_length]) * 100
        return roc

    def percentage_price_oscillator(self, close_prices):
        ema_12 = self.exponential_moving_average(close_prices, 12)
        ema_26 = self.exponential_moving_average(close_prices, 26)
        ppo = ((ema_12 - ema_26) / ema_26) * 100
        signal_line = self.exponential_moving_average(ppo, 9)
        return ppo, signal_line


    def williams_r(self, high_prices, low_prices, close_prices, window_length=14):
        if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices):
            raise ValueError("The lengths of high prices, low prices and close prices must be equal.")
        if len(high_prices) < window_length:
            raise ValueError("The length of prices must be greater than the window length.")
        wr = np.zeros(len(high_prices))
        for i in range(window_length, len(high_prices)):
            highest_high = np.max(high_prices[i-window_length:i])
            lowest_low = np.min(low_prices[i-window_length:i])
            wr[i] = ((highest_high - close_prices[i]) / (highest_high - lowest_low + self._lambda)) * -100
        return wr
    
class DatasetProcessing():
     def __init__(self):
          #super().__init__()
          self._lambda = 22e-12
            
     def norm_minmax(self, x_data, minimum=-1, maximum=1, axis=None):
        if axis is None:
            axis = self.axis

        # Verificar valores inválidos no input
        if np.isnan(x_data).any():
            raise ValueError("x_data has NaN values.")
            
        elif np.isinf(x_data).any():
            inf_indices = np.where(np.isinf(x_data))  # Localiza os índices onde há infinito
            print(f"Valores infinitos encontrados em x_data nas posições: {inf_indices}")
            print(f'shape: {x_data.shape}')
            print(f'Xdata: {x_data}')

        
        # Calcular min e max
        samples_min = np.min(x_data, axis=axis, keepdims=True)
        samples_max = np.max(x_data, axis=axis, keepdims=True)

        # Evitar divisão por zero
        range_diff = samples_max - samples_min
        range_diff[range_diff == 0] = self._lambda  # Substituir 0 por lambda

        # Normalizar
        x_data = (x_data - samples_min) * (maximum - minimum) / range_diff + minimum

        return x_data
     
     def apply_NomrMinmax(self, features, min_norm, max_norm, axis=0):
        if axis is None:
            axis = self.axis

        norm_features=np.zeros_like(features)
        for idx in range(len(features)):
            norm_features[idx]= self.norm_minmax(features[idx], minimum= min_norm, maximum= max_norm, axis=axis)
        return norm_features


     def split_data(self, X : np.array , date_time : np.datetime64, factor=0.70):
          """Split the data in train validation or test

          Args:
               X (np.array): _description_
               y (np.array): _description_
               date_time (np.datetime64): _description_
               factor (float, optional): _description_. Defaults to 0.70.

          Returns:
               _type_: _description_
          """
          nits=round(len(X)*factor)

          X_train=X[:nits]

          nit_test= np.max(X_train.shape) -1
          X_test = X[nit_test:]

          T_train = date_time[:nits]
          T_test = date_time[nit_test:]
          
          return X_train,X_test, T_train, T_test

     def weighted_categorical_crossentropy(self,weights):
          """
          from https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
          A weighted version of keras.objectives.categorical_crossentropy
          
          Variables:
               weights: numpy array of shape (C,) where C is the number of classes
          
          Usage:
               weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
               loss = weighted_categorical_crossentropy(weights)
               model.compile(loss=loss,optimizer='adam')
          """
          
          #weights = K.variable(weights)
          weights = tf.Variable(weights, dtype=tf.float32)    
          def loss(y_true, y_pred):
               # scale predictions so that the class probas of each sample sum to 1
               #y_true_printed = tf.print("y_true =", y_true)
               #y_pred_printed = tf.print("y_pred =", y_pred)
               
               y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
               # clip to prevent NaN's and Inf's
               y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
               # calc
               loss = y_true * K.log(y_pred) * weights
               loss = -K.sum(loss, -1)
               return loss
          
          return loss
     def augment_data(self, features, y_output, target_class_counts):  
            Y_categorical=np.argmax(y_output, axis=1)

            augmented_features = []
            augmented_output = []

            augmented_features.append(features)
            augmented_output.append(y_output)

            for label, new_count in  target_class_counts.items():
                idxs= Y_categorical == label

                each_features =features[idxs]
                each_labels=y_output[idxs]

                target_count =len(each_labels)
                if target_count < new_count:
                    augmented_class_data, augmented_class_labels = resample(
                        each_features, each_labels,
                        replace=True,  # Permitir repetição
                        n_samples=new_count - target_count,  # Adicionar exemplos
                        random_state=42
                    )

                    augmented_features.append(augmented_class_data)
                    augmented_output.append(augmented_class_labels)

            augmented_features = np.vstack(augmented_features)
            augmented_output = np.vstack(augmented_output)
            
            return augmented_features, augmented_output
    

class FeaturesDataGenerator(DatasetProcessing, ComputIndicators, Sequence):

    def __init__(self, X_df = None, datatype='1D', lookback=1, pred_days=1, buy_sell_threshold=[0.05,-0.05], axis=0, batch_size=32, shuffle=False, processing=None, selected_features= None, data_augmentation=False, min_max_norm=[0,1]):
        """
        Args:
            Features dataset_generator: The dataset generator providing input and output data.
            axis (int): Axis for feature computation.
            processing: Optional data processing function.
            selected_features (list): List of features to compute. If None, compute all features.
        """
        if  isinstance(X_df, pd.DataFrame): 
            X_data = X_df
        else :
            X_df = pd.DataFrame(data=np.ones([20,6]),columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
            X_data = X_df

        self.X_df = X_df
        self.InputData = X_data
        self.lookback = lookback
        self.pred_days= pred_days
        self.inputShape = X_data.shape
        #self.outputShape = dataset_generator.__getitem__(0)[1].shape
        self.processing = processing
        self.axis = axis
        self.selected_features = selected_features
        self.data_augmentation = data_augmentation
        self.min_norm=min_max_norm[0]
        self.max_norm=min_max_norm[1]
        self.datatype = datatype
        self._lambda = 22e-12
        #self.y_classification = self.comput_outputs(self.features[:,lookback-1])
        #self.y_classification = self.comput_outputs(self.InputData['Close'], days_lookback = self.pred_days)
        self.y_classification = self.label_data(close_prices=self.InputData['Close'].values, window=self.pred_days, positive_threshold=buy_sell_threshold[0], negative_threshold=buy_sell_threshold[1])[self.lookback:]
        print('self.pred_days', self.pred_days)
        self.features = self.comput_features(np.squeeze(self.InputData), pred_days = self.pred_days)
        
        if self.data_augmentation == True:
            #smote = SMOTE(sampling_strategy='auto', random_state=42)
            #self.features, self.y_classification = smote.fit_resample(self.features[:], self.y_classification[:])
            
            #APPLIED RANDOM OVER SAMPLER 
            #os = RandomOverSampler()
            #self.features, self.y_classification = os.fit_resample(self.features[:], self.y_classification[:])

            Y_train_categorical=np.argmax(self.y_classification, axis=1)
            classes, counts = np.unique(Y_train_categorical, return_counts=True)

            max_class = classes[np.argmax(counts)]
            max_count = np.max(counts)

            desired_count = int(0.60 * max_count)

            target_class_counts = {i: desired_count if i != max_class else max_count for i in classes}

            self.features, self.y_classification = self.augment_data(self.features, self.y_classification, target_class_counts)

            #self.InputData, self.y_classification = smote.fit_resample(self.InputData[self.lookback:].reshape(-1,1), self.y_classification)
            #self.y_classification = self.comput_outputs(self.InputData)
        
        #self.features = self.comput_features(np.squeeze(self.InputData))

        self.inputShape  = self.features.shape
        self.output_shape=self.y_classification[0].shape

        print('input data shape', self.inputShape)
        print('output data shape', self.y_classification.shape)

        self.batchSize= batch_size
        self.shuffle = shuffle

        #self.indices=np.arange(self.__len__() + self.batchSize)
        if self.batchSize>1:
            self.indices=np.arange(self.__len__() + self.batchSize)
        else:
            self.indices=np.arange(self.__len__())

        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __len__(self):

        return (np.max(self.inputShape) // self.batchSize)
    #- (len(self.selected_features)+self.lookback)
    
    def windowing(self, data, lookback= 1, pred_days = 0):
        y_window = []
        for current_k in np.arange(lookback +1, (len(data) - pred_days)+1 ,1):
            
            y_window += [data[current_k-lookback:current_k]]

        y_window = np.array(y_window)
        return y_window
    
    def diff_window_samples(self, data, days_lookback):
        # Calculate percentage variations
        diff_window= ((data - np.roll(data, -days_lookback)) / data) * 100
        variations=np.array(np.squeeze(diff_window[:-days_lookback][self.lookback:])).reshape(-1,)
        return variations
    
        #return np.squeeze(diff_window[:-self.lookback])

    def get_variations(self, data, days_lookback):
        # Calculate percentage variations
        variations=np.zeros(len(data))
        diff_window= np.array([((day - data[i -days_lookback if days_lookback > 0 else 0])/day)*100 for i, day in enumerate(data[days_lookback:])])
        variations[self.lookback:]=np.squeeze(diff_window[self.lookback:]).reshape(-1,)

        return variations

        #return np.squeeze(diff_window[:-self.lookback])

    def label_data__(self, close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
        # Initialize all labels as 'Hold'
        """
        ref: Stock Trading Classifier with Multichannel Convolutional Neural Network
        
        Data is labeled as per the logic in research paper
        params:
            close_prices => numpy array or list of close_prices to determine strategy
            window_size => the size of the moving window for labeling
        returns:
            numpy array with integer labels (1, 0, 2) for each window center
        """

        total_rows = len(close_prices)
        labels = [[1, 0, 0]] * total_rows  # [Hold, Buy, Sell]

        # Iterate through the closing prices using a sliding window
        for row in np.arange(0 , total_rows , 1):

            window_begin = row

            window_end = min(window_begin + window, total_rows)
            
            # Get the current window of prices
            prices_window = close_prices[window_begin:window_end]
        
            # Find the minimum and maximum values in the current window
            min_value = np.min(prices_window)
            max_value = np.max(prices_window)
            
            for i, price in enumerate(prices_window):
                idx = window_begin + i
                if  idx + 1 < total_rows:

                    if  price == min_value and price is not None:
                        labels[idx] = [1, 0, 0]  # Hold signal
                        labels[idx +1] = [0, 1, 0]  # Buy signal

                    elif price == max_value and price is not None:
                        labels[idx] = [1, 0, 0]  # Hold signal
                        labels[idx +1] = [0, 0, 1]  # Sell signal
                    
                    #else:
                    #    labels[idx] = [1, 0, 0]  # Hold signal

        return np.array(labels)
    

    def label_data_master(self, close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
        # Initialize all labels as 'Hold'
        labels = [[1, 0, 0]] * len(close_prices)  # [Hold, Buy, Sell]
        
        total_days = len(close_prices)
        
        # Iterate through the closing prices using a sliding window
        for win_begin in range(total_days - window + 1):
            win_end = win_begin + window
            
            # Get the current window of prices
            current_window = close_prices[win_begin:win_end]
            
            # Find the minimum and maximum values in the current window
            min_value = min(current_window)
            max_value = max(current_window)
            
            # Label the days based on the min and max values
            for i in range(win_begin, win_end):
                if close_prices[i] == min_value and close_prices[i] is not None:
                    if i + 1 < len(labels):  # Ensure we don't go out of bounds
                        labels[i+1] = [0, 1, 0]  # Buy on the next day
                elif close_prices[i] == max_value and close_prices[i] is not None:
                    if i + 1 < len(labels):  # Ensure we don't go out of bounds
                        labels[i+1] = [0, 0, 1]  # Sell on the next day

        return np.array(labels)
    
    def label_data(self, close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
        """
        ref: Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach
        
        Data is labeled as per the logic in research paper
        params:
            close_prices => numpy array or list of close_prices to determine strategy
            window_size => the size of the moving window for labeling
        returns:
            numpy array with integer labels (1, 0, 2) for each window center
        """
        total_rows = len(close_prices)
        labels = [[1, 0, 0]] * total_rows  # init all signals as Hold
        print(len(labels))
        print("Calculating labels")

        for row in np.arange( 0 , total_rows , 1):
            window_begin = row

            window_end = min(window_begin + window, total_rows)
            
            window_middle = (window_begin + window_end) // 2
            
            window_values= close_prices[window_begin:window_end]

            # find the index based in the max and min value in the window
            min_index = np.argmin(window_values) + window_begin
            max_index = np.argmax(window_values) + window_begin

            # define the label based in the min and max index
            
            if max_index  == window_middle:
                labels[window_middle] = [0,0,1]  # SELL
            elif min_index == window_middle:
                labels[window_middle] = [0,1,0]  # BUY
            else:    
                labels[window_middle] = [1,0,0]  # HOLD

        return np.array(labels)
    
    def label_data_v3(self,close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
        """
        Rotula os dados como 'BUY', 'SELL' ou 'HOLD' com base no Algorithm 1 Labelling Method.

        Parâmetros:
        - close_prices: array-like, preços de fechamento.
        - window: int, tamanho da janela de análise.

        Retorna:
        - np.array: lista de rótulos codificados como [hold, buy, sell].
        """
        labels = [[1, 0, 0]] * len(close_prices)  # Inicializa todos os rótulos como 'hold'

        for counter_row in range(len(close_prices)):
            if counter_row >= window:
                # Definir os índices da janela
                window_begin_index = counter_row - window
                window_end_index = counter_row
                window_middle_index = (window_begin_index + window_end_index) // 2

                # Extrair a janela de preços
                window_prices = close_prices[window_begin_index:window_end_index + 1]

                # Determinar o preço mínimo e máximo e seus índices
                min_value = np.min(window_prices)
                max_value = np.max(window_prices)

                min_index = np.argmin(window_prices) + window_begin_index
                max_index = np.argmax(window_prices) + window_begin_index

                # Aplicar a lógica de rotulagem
                if max_index == window_middle_index:
                    labels[window_middle_index] = [0, 0, 1]
                elif min_index == window_middle_index:
                    labels[window_middle_index] = [0, 1, 0]

        return np.array(labels)
    
    def label_data_v2(self, close_prices, window=7, positive_threshold=0.05, negative_threshold=-0.05):
        # Inicializar as variáveis
        labels = [[1, 0, 0]] * len(close_prices)  # Inicializar com "Hold"
        
        # Loop para processar as janelas de preços
        for i in range(window, len(close_prices)):
            # Calcular a variação do preço
            variation = (close_prices[i] - close_prices[i - window]) / close_prices[i - window]
            
            # Verificar se a variação está dentro dos limites
            if variation > positive_threshold:
                # Se a variação for positiva, adicionar "Buy" no índice i
                labels[i] = [0, 1, 0]  # Buy
            elif variation < negative_threshold:
                # Se a variação for negativa, adicionar "Sell" no índice i
                labels[i] = [0, 0, 1]  # Sell
        
        # Retornar as labels
        return np.array(labels)
    
    def label_data_v1(self,close_prices, window=7, positive_threshold=0.05, negative_threshold=-0.05):
        """
        Gera rótulos Buy, Sell e Hold para os preços de fechamento com base em uma janela e limiares.

        Parâmetros:
        - close_prices: array-like, preços de fechamento.
        - window: int, tamanho da janela para cálculo de máximos e mínimos.
        - positive_threshold: float, variação positiva mínima para sinal de Buy.
        - negative_threshold: float, variação negativa mínima para sinal de Sell.

        Retorno:
        - np.array: rótulos no formato [Hold, Buy, Sell].
        """
        labels = [[1, 0, 0]] * len(close_prices)  # Inicializar com "Hold" como padrão

        for i in range(len(close_prices)):
            if i + window < len(close_prices):  # Garantir que a janela não extrapole os dados
                future_window = close_prices[i:i + window]
                current_price = close_prices[i]

                # Calcular a variação percentual
                future_max = np.max(future_window)
                future_min = np.min(future_window)

                max_variation = (future_max - current_price) / current_price
                min_variation = (future_min - current_price) / current_price

                # Aplicar lógica de Buy/Sell
                if max_variation >= positive_threshold:
                    labels[i] = [0, 1, 0]  # Buy
                elif min_variation <= negative_threshold:
                    labels[i] = [0, 0, 1]  # Sell

        return np.array(labels)
    def label_data_v0(self, close_prices, window=7, positive_threshold=0.05, negative_threshold=-0.05):
        labels = []  # Store labels
        variations = []
        for i in range(len(close_prices)):
            if i + window >= len(close_prices):  # If the window exceeds data length
                labels.append([1,0,0])
                variations +=[0]
                continue

            current_price = close_prices[i]
            future_prices = close_prices[i+1:i+1+window]
            
            returns = (future_prices - current_price) / current_price


            if max(returns) >= positive_threshold:
                labels.append([0,1,0])
                variations +=[max(returns)]
            elif min(returns) <= negative_threshold:
                labels.append([0,0,1])
                variations +=[min(returns)]
            else:
                labels.append([1,0,0])
                variations +=[np.mean(returns)]
            
        self.variations = np.stack(variations)[self.lookback:]

        return np.stack(labels)


    
    def __getFeaturesName__(self):
        return self.features_name


    def __getitem__(self, idx):

        if idx == -1:
            idx = self.__len__()
            
        batch_indices = self.indices[idx : idx + self.batchSize]
        #window=len(self.selected_features)+self.lookback-1
        
        y = np.zeros([self.batchSize,self.output_shape[0]])
    
        #features = np.zeros([self.batchSize, self.features_length])
        features = np.zeros([self.batchSize, self.lookback, self.features_length])   
        for i, j in enumerate(batch_indices):
            
            #apply norm minmax for each bacth data 
            features[i,:,:] = np.nan_to_num(self.norm_minmax(self.features[j], axis=0, minimum=self.min_norm, maximum=self.max_norm))

            if np.isinf(features[i,:,:]).any():
                raise ValueError(f"Valor infinito encontrado na feature {j}. idx: {j}, Valor: {self.features[j]}")

            if np.isnan(features[i,:,:]).any():
                raise ValueError(f"Valor NaN encontrado na feature {j}. idx: {j}, Valor: {self.features[j]}")
            #features[i,:,:] = self.features[j]
            y[i,:] = self.y_classification[j]
        
        
        if self.datatype == '2D':
            # Transformar para formato 2D
            features = np.transpose(features, [0, 2, 1]).reshape(-1, 1, self.features_length, self.lookback)

        # Convertendo features e y para tensores do TensorFlow
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        return features, y
    
    def bat_data(self,x_data):
        x_data= (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
        return x_data

    def on_epoch_end(self):
        """Override the superclass method to shuffle the data on the end of the epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def getitem(self, index):
        """Public method to retrieve the batches during the training

        Args:
            index (int): batch indexs
        Returns:
            tuple[ndarray, ndarray]: (x[samples, ch], y[class]) data
        """
        return self.__getitem__(index)
    
    
    def comput_features(self, x_data, pred_days = 0 ):
        """
        Args:
            emg_data (numpy.ndarray): Input EMG data.

        Returns:
            numpy.ndarray: Feature matrix computed from the input data.
        """
        
        # list with the all features. Bag of features -pred_days*2 if pred_days > 0 else None
        prediction_horizon = -pred_days*2 if pred_days > 0 else None
        prediction_horizon = None
        all_features = {
            'Data_lookback': self.windowing(x_data['Close'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            'Close': self.windowing(x_data['Close'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            'Open': self.windowing(x_data['Open'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            'High': self.windowing(x_data['Volume'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            'Low': self.windowing(x_data['Low'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            #'Adj Close': self.windowing(x_data['Adj Close'].values.astype(np.float32),lookback = self.lookback, pred_days = pred_days*2),
            'Volume': self.windowing(x_data['Volume'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            'Volume_log': np.log(self.windowing(x_data['Volume'].values.astype(np.float32), lookback = self.lookback, pred_days = 0)),
            #'Open': x_data['Open'].values[self.lookback:prediction_horizon],
            #'High': x_data['High'].values[self.lookback:prediction_horizon],
            #'Low': x_data['Low'].values[self.lookback:prediction_horizon],
            #'Adj Close': x_data['Adj Close'].values[self.lookback:prediction_horizon], 
            #'Volume': x_data['Volume'].values[self.lookback:prediction_horizon],

            'EMA9': self.windowing(self.exponential_moving_average(x_data['Close'].values.astype(np.float32), window_length=9)[:],lookback = self.lookback, pred_days = 0), 
            'EMA20': self.windowing(self.exponential_moving_average(x_data['Close'].values.astype(np.float32), window_length=20)[:],lookback = self.lookback, pred_days = 0), 
            'EMA50': self.windowing(self.exponential_moving_average(x_data['Close'].values.astype(np.float32), window_length=50)[:],lookback = self.lookback, pred_days = 0), 
            'EMA100': self.windowing(self.exponential_moving_average(x_data['Close'].values.astype(np.float32), window_length=100)[:],lookback = self.lookback, pred_days = 0),  
            'EMA200': self.windowing(self.exponential_moving_average(x_data['Close'].values.astype(np.float32), window_length=200)[:],lookback = self.lookback, pred_days = 0), 
            'MA111': self.windowing(self.moving_average(x_data['Close'].values.astype(np.float32), window_length=111)[:],lookback = self.lookback, pred_days = 0),  
            'MA350': self.windowing(self.moving_average(x_data['Close'].values.astype(np.float32), window_length=350)[:],lookback = self.lookback, pred_days = 0),
            'MACD': self.windowing(self.macd(x_data['Close'].values.astype(np.float32))[0][:],lookback = self.lookback, pred_days = 0),  
            'MACD_Signal': self.windowing(self.macd(x_data['Close'].values.astype(np.float32))[1][:],lookback = self.lookback, pred_days = 0),  
            'MACD_Histogram': self.windowing(self.macd(x_data['Close'].values.astype(np.float32))[2][:],lookback = self.lookback, pred_days = 0),  
            'RSI_14': self.windowing(self.rsi(x_data['Close'].values.astype(np.float32), period=14)[:],lookback = self.lookback, pred_days = 0),
            'CCI': self.windowing(self.cci(x_data['High'].values.astype(np.float32), x_data['Low'].values.astype(np.float32), x_data['Close'].values.astype(np.float32))[:],lookback = self.lookback, pred_days = 0),
            'Stochastic_K': self.windowing(self.stochastic(x_data['High'].values.astype(np.float32), x_data['Low'].values.astype(np.float32), x_data['Close'].values.astype(np.float32))[0][:],lookback = self.lookback, pred_days = 0),
            'Stochastic_D': self.windowing(self.stochastic(x_data['High'].values.astype(np.float32), x_data['Low'].values.astype(np.float32), x_data['Close'].values.astype(np.float32))[1][:],lookback = self.lookback, pred_days = 0),
            'Bollinger_Bands_Upper': self.windowing(self.bollinger_bands(x_data['Close'].values.astype(np.float32))[0][:],lookback = self.lookback, pred_days = 0),
            'Bollinger_Bands_Middle': self.windowing(self.bollinger_bands(x_data['Close'].values.astype(np.float32))[1][:],lookback = self.lookback, pred_days = 0),
            'Bollinger_Bands_Lower': self.windowing(self.bollinger_bands(x_data['Close'].values.astype(np.float32))[2][:],lookback = self.lookback, pred_days = 0),
            'variations': self.windowing(self.get_variations(x_data['Close'].values.astype(np.float32), days_lookback = 0),lookback = self.lookback, pred_days = 0),
            'Chaikin_Money_Flow': self.windowing(self.chaikin_money_flow(x_data['High'].values.astype(np.float32), x_data['Low'].values.astype(np.float32), x_data['Close'].values.astype(np.float32), x_data['Volume'].values.astype(np.float32))[:],lookback = self.lookback, pred_days = 0),
            'Williams_R': self.windowing(self.williams_r(x_data['High'].values.astype(np.float32), x_data['Low'].values.astype(np.float32), x_data['Close'].values.astype(np.float32), self.lookback)[:],lookback = self.lookback, pred_days = 0),
            'ROC': self.windowing(self.rate_of_change(x_data['Close'].values.astype(np.float32))[:],lookback = self.lookback, pred_days = 0),
            'PPO': self.windowing(self.percentage_price_oscillator(x_data['Close'].values.astype(np.float32))[:],lookback = self.lookback, pred_days = 0),
            'SCP': self.windowing(self.SCP (x_data['Close'].values.astype(np.float32))[:],lookback = self.lookback, pred_days = 0),
            'MFI': self.windowing(self.money_flow_index(x_data['High'].values.astype(np.float32), x_data['Low'].values.astype(np.float32), x_data['Close'].values.astype(np.float32), x_data['Volume'].values.astype(np.float32))[:],lookback = self.lookback, pred_days = 0)

        }
                               
        if self.selected_features is None:
            selected_features = [key for key in all_features.keys()]
            
        else:
            selected_features = self.selected_features

        if 'Data_lookback' in selected_features:
            _selected_features = selected_features.copy()
            _selected_features.remove('Data_lookback')

            Data_lookback = np.vstack(all_features['Data_lookback'])

            if len(_selected_features) == 0:
                features = Data_lookback
            else:
                
                #for feature in _selected_features:
                #    print(feature, all_features[feature].shape)
                
                
                features_1d = [feature for feature in [all_features[feature] for feature in _selected_features] if feature.ndim == 1]
                features_2d = [feature for feature in [all_features[feature] for feature in _selected_features] if feature.ndim == 2]

                #features_1d = np.array(features_1d).T
                #features = np.concatenate(features_2d + [features_1d], axis=1)
                #features = np.concatenate([Data_lookback,features],axis=1)

                features = np.concatenate(( Data_lookback[np.newaxis,:,:], features_2d), axis=0)
                features=features.transpose(1,2,0)
        else:
            #features = np.hstack([all_features[feature][:, np.newaxis] for feature in selected_features])
            #features_1d = np.array([feature for feature in [feature for feature in selected_features] if all_features[feature].ndim != 2])
            features_2d = np.array([feature for feature in [all_features[feature] for feature in selected_features] if feature.ndim == 2])
            features=features_2d.transpose(1,2,0)

        
        self.features_name = selected_features

        if self.processing is not None:
            features, _ = self.processing(features, features)

        self.features_length= features.shape[2]
        return features[:]

class scrapingHistoricalData:
    def __init__(self):
        pass
    def get_crypto_historical_data(self, cryptos, interval='1d', start_time='2017-01-01', end_time=None):
        """
        Obtem dados históricos da Binance API
        
        Parameters:
        cryptos (list): Lista de símbolos de criptomoedas (ex.: 'BTC', 'ETH', etc.)
        interval (str): Intervalo de tempo (ex.: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
        start_time (str): Data de início para buscar dados (formato 'YYYY-MM-DD')
        
        Returns:
        DataFrame: Dados históricos das criptomoedas
        
        """
        # Adiciona o par USD para cada cripto
        cryptos = [crypto + "USDT" for crypto in cryptos]
        
        # Função para baixar dados históricos da Binance API

        
        # Baixar e consolidar dados em um DataFrame
        cryptos_df = pd.DataFrame()
        for crypto in cryptos:
            data = self.get_binance_data(crypto, interval, start_time, end_time)
            # Ajusta o nome da coluna removendo "USDT" antes de adicionar ao DF
            #data.columns = [crypto.replace('USDT', '')]
            if cryptos_df.empty:
                cryptos_df = data
            else:
                cryptos_df = pd.concat([cryptos_df, data], axis=1)
        
        if not cryptos_df.empty:
            cryptos_df = cryptos_df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            })
            # Visualização dos dados
            cryptos_df = cryptos_df.rename_axis('Date')
            cryptos_df.reset_index(inplace= True)
        return cryptos_df
    
    def get_binance_data(self, symbol, interval, start_time, end_time=None):
        base_url = "https://api.binance.com/api/v3/klines"

        if end_time == None:
            end_time = int(pd.Timestamp.now().timestamp() * 1000)  # Data atual
        
        else:
            end_time = int(pd.Timestamp(end_time).timestamp() * 1000)

        # Converte a data de início para timestamp
        start_timestamp = int(pd.Timestamp(start_time).timestamp() * 1000)
        
        # Lista para armazenar os dados
        data_list = []
        
        # Faça requisições iterativas até obter todos os dados
        while start_timestamp < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_timestamp,
                'endTime': start_timestamp + (1000 * 60 * 60 * 24 * 183),  # Intervalo de 6 meses
                'limit': 1000  # Máximo de registros por chamada
            }
            
            # Coleta dados da API
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Verifica se a resposta da API está vazia
            if data:
                # Converte os dados em DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # Adiciona os dados à lista
                data_list.append(df)
            
            # Atualiza o start_timestamp para a próxima requisição
            start_timestamp += (1000 * 60 * 60 * 24 * 183)  # Adiciona 6 meses ao timestamp
        
        # Concatena todos os DataFrames
        if data_list:
            df = pd.concat(data_list, ignore_index=True)
            
            # Formata e filtra os dados necessários
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['close'] = df['close'].astype(float)
            df['low'] = df['low'].astype(float)
            df['high'] = df['high'].astype(float)
            df['open'] = df['open'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            return df
        else:
            print("Não há dados disponíveis.")
            return pd.DataFrame()