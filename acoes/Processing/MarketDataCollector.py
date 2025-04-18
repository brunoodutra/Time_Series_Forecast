import pandas as pd
import requests

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