import numpy as np

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
            
            multiplier = ((close - low) - (high - close)) / (high - low + self._lambda)
            money_flow_volume = volume * multiplier 
            
            multipliers[i] = multiplier
            money_flow_volumes[i] = money_flow_volume
        
        cmf = np.zeros(len(high_prices))
        for i in range(window_length, len(high_prices)):
            window_slice = money_flow_volumes[i - window_length + 1: i + 1]
            volume_window_slice = volumes[i - window_length + 1: i + 1]
            cmf[i] = np.sum(window_slice) / (np.sum(volume_window_slice) +self._lambda)
    
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