from keras.utils import Sequence
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import resample

from CustomTrainLosses import CustomTrainLosses
from MarketIndicators import ComputIndicators
from MarketDataCollector import scrapingHistoricalData

import sys
class DatasetProcessing():
     def __init__(self):
          #super().__init__()
          self._lambda = 22e-12
    
     def denormalize(self, normalized_values, target_min, target_max):
        """Reverte a normalização para obter valores na escala original"""
        return normalized_values * (target_max - target_min) + target_min

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
     
     def normalize_regression(self, data, target_min, target_max):
        """Normaliza qualquer dado usando os parâmetros do treino"""
        return (data - target_min) / (target_max - target_min + 1e-8)  # +1e-8 evita 

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

     
     def augment_data(self,features, y_output, target_class_counts, shuffle=False):  
        
        stock_idx = np.arange(len(y_output)) 
        idx_buy = [i for i, vetor in enumerate(y_output.tolist()) if vetor == [0, 1, 0]]
        idx_sell = [i for i, vetor in enumerate(y_output.tolist()) if vetor == [0, 0, 1]]
        idx_hold = [i for i, vetor in enumerate(y_output.tolist()) if vetor == [1, 0, 0]]


        def resample(old_idx, new_len):
            idx_resampled=[]
            idx=0
            while len(idx_resampled) < new_len:
                idx_resampled.append(old_idx[idx])
                idx +=1
                if idx >= len(old_idx):
                    idx=0
            return idx_resampled

        new_buy_idx=resample(idx_buy, target_class_counts[1]-len(idx_buy))
        new_sell_idx=resample(idx_sell, target_class_counts[2]-len(idx_sell))
        #new_hold_idx=resample(idx_hold, target_class_counts[0])
        stock_idx = stock_idx.tolist() + new_buy_idx + new_sell_idx 

        if shuffle:
            np.random.shuffle(stock_idx)

        return features[stock_idx], y_output[stock_idx], stock_idx

     def augment_data_2(self, features, y_output, target_class_counts):  
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
    

class FeaturesDataGenerator(ComputIndicators, DatasetProcessing, CustomTrainLosses, Sequence):

    def __init__(self, X_df = None, datatype='1D', predict_type='classification', lookback=1, pred_days=1, buy_sell_threshold=[0.05,-0.05], axis=0, batch_size=32, shuffle=False, processing=None, selected_features= None, data_augmentation=False, 
                 min_max_norm_features=[0,1], min_max_regression_targets=None):
        """
        Args:
            Features dataset_generator: The dataset generator providing input and output data.
            axis (int): Axis for feature computation.
            processing: Optional data processing function.
            selected_features (list): List of features to compute. If None, compute all features.
        """
        
        #self.InputData = X_df
        self.lookback = lookback
        self.pred_days= pred_days
        #self.inputShape = X_data.shape
        #self.outputShape = dataset_generator.__getitem__(0)[1].shape
        self.processing = processing
        self.axis = axis
        self.selected_features = selected_features
        self.data_augmentation = data_augmentation
        self.min_max_norm_features =  min_max_norm_features
        self.min_norm=min_max_norm_features[0]
        self.max_norm=min_max_norm_features[1]
        self.min_max_regression_targets=min_max_regression_targets
        self.datatype = datatype
        self._lambda = 22e-12
        self.predict_type = predict_type
        #self.y_classification = self.comput_outputs(self.features[:,lookback-1])
        #self.y_classification = self.comput_outputs(self.InputData['Close'], days_lookback = self.pred_days)
        if  isinstance(X_df, pd.DataFrame):
            X_df=[X_df]
        elif X_df == [] or X_df==None:
            X_df=[None]


        self.features = []
        self.y_classification = []
        self.y_regression = []
        for X_data in X_df:

            if  isinstance(X_data, pd.DataFrame): 
                InputData = X_data
            else :
                X_df = pd.DataFrame(data=np.ones([20,6]),columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
                InputData = X_df

            y_classification = self.label_data(close_prices=InputData['Close'].values, 
                                                    window=self.pred_days, 
                                                    positive_threshold=buy_sell_threshold[0], 
                                                    negative_threshold=buy_sell_threshold[1])[self.lookback:]
            #print('self.pred_days', self.pred_days)
            features = self.comput_features(np.squeeze(InputData), pred_days = self.pred_days)

            self.window_close_values =  self.windowing(InputData['Close'].values.astype(np.float32), lookback = 1, pred_days = 0)[lookback-1:]

            if self.data_augmentation == True:
                
                #APPLIED OVER SAMPLER 
                    # over sample repeating the historical data 
                Y_train_categorical=np.argmax(y_classification, axis=1)
                classes, counts = np.unique(Y_train_categorical, return_counts=True)

                max_class = classes[np.argmax(counts)]
                max_count = np.max(counts)
                
                #using 60% of desbalance
                desired_count = int(0.60 * max_count) #int(0.60 * max_count)

                # get the classes counts
                target_class_counts = {i: desired_count if i != max_class else max_count for i in classes}

                features, y_classification, stock_idx = self.augment_data(features, y_classification, [max_count, int(max_count*0.6), int(max_count*0.6)])

                self.window_close_values=self.window_close_values[stock_idx]
            

            if self.min_max_regression_targets == None:
                self.target_min = np.min(self.window_close_values)
                self.target_max = np.max(self.window_close_values)
            else:
                self.target_min = self.min_max_regression_targets[0]
                self.target_max = self.min_max_regression_targets[1]

            #TODO: verify if it make senses
            #y_regression = self.normalize_regression(self.window_close_values, self.target_min, self.target_max)
            y_regression = self.window_close_values
            
            self.features += [features]
            self.y_classification += [y_classification]
            self.y_regression += [y_regression]
        
        self.features = np.vstack(self.features)
        self.y_classification =  np.vstack(self.y_classification)
        self.y_regression = np.vstack(self.y_regression)

        self.inputShape  = self.features.shape
        self.output_shape=self.y_classification[0].shape

        print('input data shape', self.inputShape)
        print('output data shape', self.y_classification.shape)

        self.batchSize= batch_size
        self.shuffle = shuffle

        #self.indices=np.arange(self.__len__() + self.batchSize)
        #if self.batchSize>1:
        #    self.indices=np.arange(self.__len__() + self.batchSize)
        #else:
        #    self.indices=np.arange(self.__len__())

        self.indices=np.arange(self.inputShape[0])

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

    def label_data_master__(self, close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
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
        #labels = [[1, 0, 0]] * total_rows  # [Hold, Buy, Sell]
        labels = [[1, 0, 0] for _ in range(total_rows)]  # Correção: listas independentes

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
                        #labels[idx] = [1, 0, 0]  # Hold signal
                        labels[idx] = [0, 1, 0]  # Buy signal

                    elif price == max_value and price is not None:
                        #labels[idx] = [1, 0, 0]  # Hold signal
                        labels[idx] = [0, 0, 1]  # Sell signal
                    
                    else:
                        labels[idx] = [1, 0, 0]  # Hold signal

        return np.array(labels)
    
    def label_data(self, close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
        
        total_days = len(close_prices)
        labels = [[1, 0, 0] for _ in range(len(close_prices))]  # Correção: listas independentes
        winBegin = 0
        winEnd = winBegin + window
        countRow = 0

        while countRow <= total_days:
             # Get the current window of prices
            current_window = close_prices[winBegin:winEnd]

            minValue = min(current_window)
            maxValue = max(current_window)

            for i in range(winBegin, winEnd):
                if close_prices[i] == minValue and close_prices[i] is not None:
                    labels[i-1] = [1,0,0]
                    labels[i] = [0,1,0]
                elif close_prices[i] == maxValue and close_prices[i] is not None:
                    labels[i-1] = [1,0,0]
                    labels[i] = [0,0,1]
                elif close_prices[i] is not None:
                    labels[i] = [1,0,0]

            winBegin = winEnd + 1
            winEnd = winBegin + window
            countRow = winEnd

        return np.array(labels)

    def label_data_master(self, close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
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
        #labels = [[1, 0, 0]] * total_rows  # init all signals as Hold
        labels = [[1, 0, 0] for _ in range(total_rows)]  # Correção: listas independentes

        print(len(labels))
        print("Calculating labels")
        countRow = 0
        while countRow <= total_rows:
            countRow += 1 
            if countRow > window:
                window_begin = countRow - window
                window_end = window_begin + window
                window_middle_index = (window_begin + window_end) // 2
                
                if window_end > total_rows:
                    pass

                else:
                    window_values= close_prices[window_begin:window_end]

                    # find the index based in the max and min value in the window
                    min_index = np.argmin(window_values) + window_begin
                    max_index = np.argmax(window_values) + window_begin

                    # define the label based in the min and max index
                    
                    if max_index  == window_middle_index:
                        labels[window_middle_index] = [0,0,1]  # SELL
                    elif min_index == window_middle_index:
                        labels[window_middle_index] = [0,1,0]  # BUY
                    else:    
                        labels[window_middle_index] = [1,0,0]  # HOLD

        return np.array(labels)
    
    def label_data__(self, close_prices, window=11, positive_threshold=0.05, negative_threshold=-0.05):
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

            #window_end = min(window_begin + window, total_rows)
            window_end = window_begin + window

            if window_end > total_rows:
                pass

            else:
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
        
        batch_idx=idx*self.batchSize + (idx-1)
        end = batch_idx + self.batchSize

        if end > len(self.indices):
            end = len(self.indices)
            batch_indices = self.indices[batch_idx:end]
        else:
            batch_indices = self.indices[batch_idx:end]
            

        #window=len(self.selected_features)+self.lookback-1
        
        y_class = np.zeros([self.batchSize,self.output_shape[0]])

        y_regression = np.zeros([self.batchSize,1])

        #features = np.zeros([self.batchSize, self.features_length])
        features_norm = np.zeros([self.batchSize, self.lookback, self.features_length])  
        features = np.zeros([self.batchSize, self.lookback, self.features_length])    
        for i, j in enumerate(batch_indices):
            
            #apply norm minmax for each bacth data 
            features_norm[i,:,:] = np.nan_to_num(self.norm_minmax(self.features[j].copy(), axis=0, minimum=self.min_norm, maximum=self.max_norm))

            if np.isinf(features_norm[i,:,:]).any():
                raise ValueError(f"Valor infinito encontrado na feature {j}. idx: {j}, Valor: {self.features[j]}")

            if np.isnan(features_norm[i,:,:]).any():
                raise ValueError(f"Valor NaN encontrado na feature {j}. idx: {j}, Valor: {self.features[j]}")
            
            features[i,:,:] = self.features[j].copy()
            
            y_regression[i,:] = self.y_regression[j]  # Use the normalized value 
    

            y_class[i,:] = self.y_classification[j]
        
        if self.datatype == '2D':
            # Transformar para formato 2D
            features_norm = np.transpose(features_norm, [0, 2, 1]).reshape(-1, self.lookback, self.features_length, 1)

        # Convertendo features e y para tensores do TensorFlow
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        features_norm = tf.convert_to_tensor(features_norm, dtype=tf.float32)
        y_class = tf.convert_to_tensor(y_class, dtype=tf.float32)
        y_regression = tf.convert_to_tensor(y_regression, dtype=tf.float32)

        if self.predict_type =="classification":
            return features_norm, y_class
        elif self.predict_type =="regression":
            return features, y_regression
        elif self.predict_type =="both":
            return features_norm, [y_regression,y_class]
    
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
    
    
    def comput_features(self, x_data, pred_days = 0):
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
            'High': self.windowing(x_data['High'].values.astype(np.float32), lookback = self.lookback, pred_days = 0),
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

