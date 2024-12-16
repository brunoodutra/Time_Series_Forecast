from tqdm import tqdm
import time
from keras.utils import Sequence
from keras import backend as K
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler 

class ComputIdicators():
     def __init__(self):
          pass
     def autorregressive_coefs(self,data,p=3):
        channels=data.shape(self.axis) 

        phi=np.zeros([data.shape[0],data.shape[1],p])

        ar_coefs=np.zeros([channels,p])

        y_init=[]

        y = data

        aux=np.zeros(p)
        for k in range(p): 
            aux[p-k:]=y[:k]
            y_init.append(aux.copy())

        phi=np.vstack([y[i-p:i] if i-p>=0 else y_init[i]  for i in range(0, len(y))])

        ar_coefs[:]=np.linalg.inv(phi[:].T.dot(phi[:])).dot(phi[:].T.dot(data[:]))
            
        return ar_coefs
     
     def moving_average(self,data, window_length):
          """
          Calculates the Moving Average (MA) of a time series data.

          Args:
               data: A list or NumPy array containing the time series data.
               window_length: The number of periods to use for the MA calculation (window size).

          Returns:
               A NumPy array containing the MA values for each data point.
          """

          if window_length < 1:
               raise ValueError("Window length must be a positive integer.")

          # Initialize empty array for MA values
          moving_average_values = np.zeros(len(data))

          # Iterate through the data
          for i in range(len(data)):
               # Check if the window goes beyond the data boundary
               if i < window_length - 1:
                    # If within the initial window, use the average of available data points
                    moving_average_values[i] = np.mean(data[:i+1])
               else:
                    # For other points, use the average of the window
                    window_slice = data[i - window_length + 1 : i + 1]
                    moving_average_values[i] = np.mean(window_slice)

          return moving_average_values
     def exponential_moving_average(self, data, window_length):
          """
          Calculates the Exponential Moving Average (EMA) of a time series data.

          Args:
               data: A list or NumPy array containing the time series data.
               window_length: The number of periods to use for the EMA calculation.

          Returns:
               A NumPy array containing the EMA values for each data point.
          """

          if window_length < 1:
               raise ValueError("Window length must be a positive integer.")

          ema = np.zeros(len(data))
          # Handle the initial EMA calculation (use simple average for the first window_length elements)
          ema[:window_length] = np.mean(data[:window_length])
          alpha = 2 / (1 + window_length)  # Smoothing factor (weight for the current data point)

          for i in range(window_length, len(data)):
               ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
          
          return ema

     def macd(self,data, fast_period=12, slow_period=26, signal_period=9):
          """
          Calculates the MACD (Moving Average Convergence Divergence) indicator for a time series data.

          Args:
               data: A list or NumPy array containing the closing price data.
               fast_period: The number of periods for the fast Exponential Moving Average (EMA). (default: 12)
               slow_period: The number of periods for the slow Exponential Moving Average (EMA). (default: 26)
               signal_period: The number of periods for the EMA of the MACD difference. (default: 9)

          Returns:
               A tuple containing three NumPy arrays:
                    - macd: The MACD line (fast EMA minus slow EMA)
                    - macd_signal: The signal line (EMA of the MACD difference)
                    - macd_histogram: The MACD histogram (MACD minus signal line)
          """

          # Check if window lengths are positive integers
          if any(period < 1 for period in [fast_period, slow_period, signal_period]):
               raise ValueError("Window lengths must be positive integers.")

          # Calculate the fast EMA and slow EMA
          ema_fast = self.exponential_moving_average(data, fast_period)
          ema_slow = self.exponential_moving_average(data, slow_period)

          # Calculate the MACD line (fast EMA minus slow EMA)
          macd = ema_fast - ema_slow

          # Calculate the MACD signal line (EMA of the MACD difference)
          macd_signal = self.exponential_moving_average(macd, signal_period)

          # Calculate the MACD histogram (MACD minus signal line)
          macd_histogram = macd - macd_signal

          return macd, macd_signal, macd_histogram

     def rsi(self, data, period=14, pred_days = 1):
          if period < 1:
               raise ValueError("Period must be a positive integer.")

          delta=self.get_variations(data, days_lookback = pred_days)
          # Calcular as diferenças dos preços
          #delta = np.diff(data)

               # Separar os ganhos e perdas
          up_changes = np.where(delta > 0, delta, 0)
          down_changes = np.where(delta < 0, np.abs(delta), 0)

          # Calcular a média dos ganhos e perdas
          avg_gain = np.zeros(len(up_changes))
          avg_loss = np.zeros(len(down_changes))

          # Calcular a média para os primeiros 'period' dados
          avg_gain[:period] = np.cumsum(up_changes[:period]) / np.arange(1, period + 1)
          avg_loss[:period] = np.cumsum(down_changes[:period]) / np.arange(1, period + 1)

          # Calcular a média para o resto dos dados
          for i in range(period, len(up_changes)):
               avg_gain[i] = (avg_gain[i - 1] * (period - 1) + up_changes[i]) / period
               avg_loss[i] = (avg_loss[i - 1] * (period - 1) + down_changes[i]) / period

          # Calcular o Relative Strength (RS)
          rs = avg_gain / avg_loss
          rs = np.where(avg_loss == 0, np.inf, rs)

          # Calcular o RSI
          rsi = 100 - (100 / (1 + rs))

          # Retornar os valores de RSI, começando do período especificado
          #print('data',data.shape)
          return np.array(np.zeros(period - 1).tolist() + rsi[period - 1:].tolist())


     def cci(self, high_prices, low_prices, close_prices, window_length=20):
          """
          Calculates the Commodity Channel Index (CCI) indicator for a time series data.

          Args:
               high_prices: A list or NumPy array containing the high prices.
               low_prices: A list or NumPy array containing the low prices.
               close_prices: A list or NumPy array containing the closing prices.
               window_length: The number of periods to use for the calculation. (default: 20)

          Returns:
               A NumPy array containing the CCI values for each data point.
          """
          typical_prices = (high_prices + low_prices + close_prices) / 3
          sma_typical_prices = self.moving_average(typical_prices, window_length=window_length)
          mean_deviation = np.mean(np.abs(typical_prices - sma_typical_prices))
          cci_values = (typical_prices - sma_typical_prices) / (0.015 * mean_deviation)
          
          return cci_values
     def stochastic(self, high_prices, low_prices, close_prices, window_length=14, smooth_k=3, smooth_d=3):
          """
          Calculates the Stochastic Oscillator indicator for a time series data.

          Args:
               high_prices: A list or NumPy array containing the high prices.
               low_prices: A list or NumPy array containing the low prices.
               close_prices: A list or NumPy array containing the closing prices.
               window_length: The number of periods to use for the calculation. (default: 14)
               smooth_k: The number of periods for smoothing %K line. (default: 3)
               smooth_d: The number of periods for smoothing %D line. (default: 3)

          Returns:
               A tuple containing two NumPy arrays:
                    - percent_k: The %K line.
                    - percent_d: The %D line.
          """

          if window_length < 1 or smooth_k < 1 or smooth_d < 1:
               raise ValueError("Window lengths must be positive integers.")

          # Calculate %K
          lowest_low = self.minimum(low_prices, window_length)
          highest_high = self.maximum(high_prices, window_length)
          percent_k = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))

          # Smooth %K to get %D
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
     import numpy as np

class ComputIndicators():
    def __init__(self):
        pass

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
        rs = avg_gain / avg_loss
        rs = np.where(avg_loss == 0, np.inf, rs)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def cci(self, high_prices, low_prices, close_prices, window_length=20):
        typical_prices = (high_prices + low_prices + close_prices) / 3
        sma_typical_prices = self.moving_average(typical_prices, window_length)
        mean_deviation = np.mean(np.abs(typical_prices - sma_typical_prices))
        cci_values = (typical_prices - sma_typical_prices) / (0.015 * mean_deviation)
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
        if window_length < 1:
            raise ValueError("Window length must be a positive integer.")
        money_flow = np.zeros(len(high_prices))
        for i in range(len(high_prices)):
            money_flow[i] = ((high_prices[i] + low_prices[i] + close_prices[i]) / 3) * volumes[i]
        positive_flow = np.zeros(len(high_prices))
        negative_flow = np.zeros(len(high_prices))
        for i in range(len(high_prices)):
            if i < window_length - 1:
                positive_flow[i] = np.sum(money_flow[:i+1][money_flow[:i+1] > 0])
                negative_flow[i] = np.sum(money_flow[:i+1][money_flow[:i+1] < 0])
            else:
                window_slice = money_flow[i - window_length + 1: i + 1]
                positive_flow[i] = np.sum(window_slice[window_slice > 0])
                negative_flow[i] = np.sum(window_slice[window_slice < 0])
        mfi = 100 - (100 / (1 + positive_flow / np.abs(negative_flow)))
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

class DatasetProcessing():
     def __init__(self):
          super().__init__()

     def norm_minmax(self, x_data, minimum=0, maximum=1, axis=None):
        """Performs the normalization of the values in x_data"""
        if axis is None:
            axis = self.axis

        samples_min = x_data.min(axis=axis, keepdims=True)
        samples_max = x_data.max(axis=axis, keepdims=True)
        
        # Verificação crucial para evitar divisão por zero
        diff = samples_max - samples_min
        mask = diff == 0  #Identifica onde a subtração resulta em 0
        diff[mask] = 1    #Substitui os zeros por 1, evitando divisão por zero

        x_data = (x_data - samples_min) * (maximum - minimum)
        x_data = (x_data / diff) + minimum

        return x_data
     
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

class FeaturesDataGenerator(DatasetProcessing, ComputIndicators, Sequence):

    def __init__(self, X_df = None, lookback=1, pred_days=1, buy_sell_threshold=[0.05,-0.05], axis=0, batch_size=32, shuffle=False, processing=None, selected_features= None, data_augmentation=False):
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
            X_df = pd.DataFrame(data=np.zeros([20,6]),columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
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
        

        #self.y_classification = self.comput_outputs(self.features[:,lookback-1])
        #self.y_classification = self.comput_outputs(self.InputData['Close'], days_lookback = self.pred_days)
        self.y_classification = self.label_data(self.InputData['Close'].values, window = self.pred_days, positive_threshold=buy_sell_threshold[0], negative_threshold=buy_sell_threshold[1])[self.lookback:]
        print('self.pred_days', self.pred_days)
        self.features = self.comput_features(np.squeeze(self.InputData), pred_days = self.pred_days)

        if self.data_augmentation == True:
            #smote = SMOTE(sampling_strategy='auto', random_state=42)
            #self.features, self.y_classification = smote.fit_resample(self.features[:], self.y_classification[:])
            
            #APPLIED RANDOM OVER SAMPLER 
            os = RandomOverSampler()
            self.features, self.y_classification = os.fit_resample(self.features[:], self.y_classification[:])

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
    def label_data(self, close_prices, window=7, positive_threshold=0.05, negative_threshold=-0.05):
        # Inicializar as variáveis
        labels = [[1, 0, 0]] * len(close_prices)  # Inicializar com "Hold"
        self.variations = self.diff_window_samples(close_prices, window)
        # Loop para processar as janelas de preços
        for i in range(len(close_prices)):
            win_begin = max(0, i - window + 1)
            win_end = i + 1
            min_value = np.min(close_prices[win_begin:win_end])
            max_value = np.max(close_prices[win_begin:win_end])

            # Verificar se o preço é o mínimo ou máximo
            if close_prices[i] == min_value:
                # Se o preço for o mínimo, adicionar "Hold" seguido de "Buy" no índice i+1
                if i + 1 < len(close_prices):
                    labels[i + 1] = [1, 0, 0]  # Hold
                    labels[i + 1] = [0, 1, 0]  # Buy
                else:
                    labels[i] = [0, 1, 0]  # Buy
            elif close_prices[i] == max_value:
                # Se o preço for o máximo, adicionar "Hold" seguido de "Sell" no índice i+1
                if i + 1 < len(close_prices):
                    labels[i + 1] = [1, 0, 0]  # Hold
                    labels[i + 1] = [0, 0, 1]  # Sell
                else:
                    labels[i] = [0, 0, 1]  # Sell

        # Retornar as labels
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
            features[i,:,:] = self.norm_minmax(self.features[j],axis=0,minimum=-1,maximum=1)
            #features[i,:,:] = self.features[j]
            y[i,:] = self.y_classification[j]

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
            'Open': self.windowing(x_data['Open'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            'High': self.windowing(x_data['Volume'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            'Low': self.windowing(x_data['Low'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            #'Adj Close': self.windowing(x_data['Adj Close'].values.astype(np.float32),lookback = self.lookback, pred_days = pred_days*2),
            'Volume': self.windowing(x_data['Volume'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
            
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
            features = np.hstack([all_features[feature][:, np.newaxis] for feature in selected_features])

        self.features_name = selected_features

        if self.processing is not None:
            features, _ = self.processing(features, features)

        self.features_length= features.shape[2]
        return features[:]