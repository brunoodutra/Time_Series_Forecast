import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go

import datetime
from pmdarima.arima import auto_arima # biblioteca que importa o arima
import pmdarima.arima as pm # biblioteca que importa o arima
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import pandas as pd

from pandas_datareader import data 
import yfinance as yfin
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, MaxPooling1D, AveragePooling1D, BatchNormalization, Flatten, Dropout, SpatialDropout1D, GRU, GlobalAveragePooling1D
from tensorflow.keras import Input

import investpy as inv

import streamlit as st

#%%
st.title('Análise e Predição de Ações')

# "with" notation
with st.sidebar:

    # Adicionar colunas para organizar os campos de data lado a lado
    col1, col2 = st.columns(2)

    # Adicionar campos para escolher o tempo de análise (início e fim)
    with col1:
        data_inicio = st.date_input("Início", pd.to_datetime('2022-01-01'))

    with col2:
        data_fim = st.date_input("Fim", pd.to_datetime('2022-12-31'))

    acoes_disponiveis =  inv.get_stocks_list("brazil")
    acoes_selecionadas = st.multiselect("Selecione até 5 ações", acoes_disponiveis, default=[])

    # Verificar se foram selecionadas até 5 ações
    if len(acoes_selecionadas) > 5:
        st.warning("Por favor, selecione no máximo 5 Ativos (opções: Ações, FIS, Cryptos).")
        acoes_selecionadas = acoes_selecionadas[:5]



# Título do aplicativo
st.title('Exemplo Análise exploratória e predição de Ativos')

#%% Carregando dados
#acoes_df=pd.read_csv('acoes.csv')

yfin.pdr_override()

acoes=acoes_selecionadas
peso_acoes=np.ones(len(acoes))*1/len(acoes)
# adc o .SA no nome de cada ação para carregar no banco de dados
for i in range(np.size(acoes)):
    acoes[i]=acoes[i]+".SA"

acoes.append('^BVSP') # inclui o índice ibovespa na ultima coluna

acoes_df = pd.DataFrame() 
for acao in acoes:
     acoes_df[acao] = data.DataReader(acao, start=data_inicio)['Close']

# substitui o .SA do nome de cada ação para visualização dos dados
for i in range(np.size(acoes)):
    acoes_df = acoes_df.rename(columns={acoes[i]:acoes[i].replace('.SA', '')})

#renomeia o índice IBOVESPA para IBOV   
acoes_df = acoes_df.rename(columns={acoes[i]:acoes[i].replace('^BVSP', 'IBOV')})

#apaga registros nulos
acoes_df.dropna(inplace=True)

acoes_df=acoes_df.reset_index()

# Exibir o dataframe
st.subheader('Dados carregados')
st.write(acoes_df)

# Exibir o dataframe
st.subheader('Correlação entre os dados')

fig, axes = plt.subplots(1, 1, figsize=(10,5))
columns=acoes_df.columns
corrmat = acoes_df[columns].corr()
mask= np.zeros_like(corrmat)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corrmat,
            vmax=1, vmin=-1,
            annot=True, annot_kws={'fontsize':15},
            cmap=sns.diverging_palette(20,220,as_cmap=True), ax=axes)
st.pyplot(fig)

#%% Carregando dados
acoes_df['Date']=pd.to_datetime(acoes_df['Date'])
acoes_df=acoes_df.set_index('Date')
for col in acoes_df.columns:
    acoes_df[col]=acoes_df[col].astype('float32')

acoes_df_normalized = acoes_df / acoes_df.iloc[0].values

# Criação do aplicativo Streamlit
st.title('Preço do Ibovespa e Ações')

# Cria um gráfico de linha usando o Plotly
fig = go.Figure()
for acoes in acoes_df.columns:
#fig.add_trace(go.Scatter(x=acoes_df.index, y=acoes_df['B3SA3'], mode='lines', name='Preço do B3SA3'))
    fig.add_trace(go.Scatter(x=acoes_df_normalized.index, y=acoes_df_normalized[acoes], mode='lines', name=f'Preço do {acoes}'))

# Configura o layout do gráfico
fig.update_layout(
    title='Preço do Ibovespa e ações',
    xaxis_title='Data',
    yaxis_title='Preço',
    showlegend=True
)

# Exibição do gráfico no Streamlit
st.plotly_chart(fig)



# Função para exibir ou ocultar as dicas
def toggle_dicas():
    if st.button("Info sobre Médias Móveis e sinais de Trades"):
        if "mostrar_dicas" not in st.session_state:
            st.session_state.mostrar_dicas = True
        else:
            st.session_state.mostrar_dicas = not st.session_state.mostrar_dicas


def exibir_dicas():
    st.markdown("""
Os swing traders geralmente utilizam médias móveis exponenciais (EMAs) de diferentes períodos para identificar tendências e sinais de compra e venda. As EMAs mais utilizadas pelos swing traders incluem:

- EMA de 9 períodos: A EMA de 9 períodos é comumente usada para identificar sinais de curto prazo e capturar movimentos rápidos do mercado. Pode fornecer sinais mais sensíveis e frequentes.

- EMA de 20 períodos: A EMA de 20 períodos é amplamente utilizada e considerada uma média móvel de curto prazo. É usada para identificar a direção da tendência de curto prazo e possíveis pontos de reversão.

- EMA de 50 períodos: A EMA de 50 períodos é frequentemente usada para identificar a direção da tendência de médio prazo. É uma média móvel amplamente observada pelos swing traders.

- EMA de 100 períodos: A EMA de 100 períodos é usada para identificar a direção da tendência de médio a longo prazo. É útil para identificar pontos de entrada e saída em operações de swing trading mais prolongadas.

- EMA de 200 períodos: A EMA de 200 períodos é uma das médias móveis mais amplamente observadas e é usada para identificar a direção da tendência de longo prazo. É frequentemente usada como um indicador-chave para determinar a tendência geral do mercado.
"""
                )
# Botão para exibir ou ocultar dicas
toggle_dicas()

# Exibir as dicas apenas se a variável de estado indicar que as dicas devem ser mostradas
if "mostrar_dicas" in st.session_state and st.session_state.mostrar_dicas:
    exibir_dicas()