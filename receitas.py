import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import streamlit as st

# Carregando o modelo e dados
st.title('Previsão de Receitas Municipais')
st.markdown('Este aplicativo tem como objetivo prever as receitas municipais do Município de Cajazeiras/PB, com base em dados históricos de arrecadação.'
             'Para isso, foi utilizado um modelo de redes neurais recorrentes (RNN) do tipo LSTM (Long Short-Term Memory).')

@st.cache_data()
def load_model_and_data():
    model_lstm = tf.keras.models.load_model('receitas_2013_2022')
    dados_receitas = pd.read_csv('data/dados_receitas.csv', sep=';')
    return model_lstm, dados_receitas

model_lstm, dados_receitas = load_model_and_data()

# Data engeeniring
dados_receitas['DATA'] = pd.to_datetime(dados_receitas['DATA'])
dados_receitas['ANO'] = dados_receitas['DATA'].dt.year
Q1 = dados_receitas['VALOR_ARRECADADO'].quantile(0.25)
Q3 = dados_receitas['VALOR_ARRECADADO'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df_sem_outliers = dados_receitas[(dados_receitas['VALOR_ARRECADADO'] >= limite_inferior) & (dados_receitas['VALOR_ARRECADADO'] <= limite_superior)]
df_sem_outliers = df_sem_outliers[['ANO', 'VALOR_ARRECADADO']]
df_sem_outliers.boxplot(by='ANO', figsize=(10, 6))
df_sem_outliers = dados_receitas[(dados_receitas['VALOR_ARRECADADO'] >= limite_inferior) & (dados_receitas['VALOR_ARRECADADO'] <= limite_superior)]
df_receitas = df_sem_outliers.copy()
df_receitas['DATA'] = pd.to_datetime(df_receitas['DATA'])
df_receitas['ANO'] = df_receitas['DATA'].dt.year
df_receitas.drop(columns=['COD_RECEITA', 'COD_CONTRIBUINTE', 'FONTE_DADOS', 'ANO'], inplace=True)
df_receitas['SMA(12)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=12).mean()
df_receitas['SMA(6)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=6).mean()
df_receitas['SMA(3)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=3).mean()
df_receitas['SMA(2)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=2).mean()
df_receitas['lag(12)'] = df_receitas['VALOR_ARRECADADO'].shift(12)
df_receitas['lag(6)'] = df_receitas['VALOR_ARRECADADO'].shift(6)
df_receitas['lag(4)'] = df_receitas['VALOR_ARRECADADO'].shift(4)
df_receitas['lag(3)'] = df_receitas['VALOR_ARRECADADO'].shift(3)
df_receitas['lag(2)'] = df_receitas['VALOR_ARRECADADO'].shift(2)
df_receitas['lag(1)'] = df_receitas['VALOR_ARRECADADO'].shift(1)
df_receitas.dropna(inplace=True)
df_receitas['DATA'] = df_receitas['DATA'].map(dt.datetime.toordinal)
df_receitas.reset_index(drop=True, inplace=True)

# Função para realizar a previsão
def predict_revenue(input_date, df_receitas, model_lstm):
    date = pd.to_datetime(input_date)
    date = date.toordinal()
    
    date_df = pd.DataFrame({'DATA': [date]})
    date_df['SMA(12)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=12).mean().iloc[-1]
    date_df['SMA(6)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=6).mean().iloc[-1]
    date_df['SMA(3)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=3).mean().iloc[-1]
    date_df['SMA(2)'] = df_receitas['VALOR_ARRECADADO'].rolling(window=2).mean().iloc[-1]
    date_df['lag(12)'] = df_receitas['VALOR_ARRECADADO'].shift(12).iloc[-1]
    date_df['lag(6)'] = df_receitas['VALOR_ARRECADADO'].shift(6).iloc[-1]
    date_df['lag(4)'] = df_receitas['VALOR_ARRECADADO'].shift(4).iloc[-1]
    date_df['lag(3)'] = df_receitas['VALOR_ARRECADADO'].shift(3).iloc[-1]
    date_df['lag(2)'] = df_receitas['VALOR_ARRECADADO'].shift(2).iloc[-1]
    date_df['lag(1)'] = df_receitas['VALOR_ARRECADADO'].shift(1).iloc[-1]
    
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    date_df_scaled = input_scaler.fit_transform(date_df)
    date_df_scaled = date_df_scaled.reshape((date_df_scaled.shape[0], 1, date_df_scaled.shape[1]))
    
    y_pred = model_lstm.predict(date_df_scaled)
    y_pred = scaler_y.inverse_transform(y_pred)
    
    return y_pred[0][0]

# Aplicando o modelo a base de dados
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X = df_receitas.drop(['VALOR_ARRECADADO'], axis=1)
y = df_receitas.loc[:, ['VALOR_ARRECADADO']]

input_scaler = scaler_x.fit(X)
output_scaler = scaler_y.fit(y)

y_norm = output_scaler.transform(y)
X_norm = input_scaler.transform(X)

X = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
y = y_norm.reshape((y_norm.shape[0], 1))

y = scaler_y.inverse_transform(y)

# Função para realizar a previsão com base na data informada
def predict_revenue(input_date, df_receitas, model_lstm):
    date = pd.to_datetime(input_date)
    date = date.toordinal()
    date_df = df_receitas.copy()
    for i in range(date - date_df['DATA'].iloc[-1]):
        row_ = pd.DataFrame(columns=date_df.columns)  # Crie um DataFrame vazio com as mesmas colunas
        date_df = pd.concat([date_df, row_], axis=0, ignore_index=True)
        date_df['SMA(12)'] = date_df['VALOR_ARRECADADO'].rolling(window=12).mean()
        date_df['SMA(6)'] = date_df['VALOR_ARRECADADO'].rolling(window=6).mean()
        date_df['SMA(3)'] = date_df['VALOR_ARRECADADO'].rolling(window=3).mean()
        date_df['SMA(2)'] = date_df['VALOR_ARRECADADO'].rolling(window=2).mean()
        date_df['lag(12)'] = date_df['VALOR_ARRECADADO'].shift(12)
        date_df['lag(6)'] = date_df['VALOR_ARRECADADO'].shift(6)
        date_df['lag(4)'] = date_df['VALOR_ARRECADADO'].shift(4)
        date_df['lag(3)'] = date_df['VALOR_ARRECADADO'].shift(3)
        date_df['lag(2)'] = date_df['VALOR_ARRECADADO'].shift(2)
        date_df['lag(1)'] = date_df['VALOR_ARRECADADO'].shift(1)
        
        # Agora, row_ terá as médias móveis calculadas até a data atual
        row_.loc[0, 'SMA(12)'] = date_df['SMA(12)'].iloc[-1]
        row_.loc[0, 'SMA(6)'] = date_df['SMA(6)'].iloc[-1]
        row_.loc[0, 'SMA(3)'] = date_df['SMA(3)'].iloc[-1]
        row_.loc[0, 'SMA(2)'] = date_df['SMA(2)'].iloc[-1]
        row_.loc[0, 'lag(12)'] = date_df['lag(12)'].iloc[-1]
        row_.loc[0, 'lag(6)'] = date_df['lag(6)'].iloc[-1]
        row_.loc[0, 'lag(4)'] = date_df['lag(4)'].iloc[-1]
        row_.loc[0, 'lag(3)'] = date_df['lag(3)'].iloc[-1]
        row_.loc[0, 'lag(2)'] = date_df['lag(2)'].iloc[-1]
        row_.loc[0, 'lag(1)'] = date_df['lag(1)'].iloc[-1]
        row_.loc[0, 'DATA'] = date_df['DATA'].iloc[-1] + 1
        
        # Resto do código permanece igual
        row_ = row_.drop(['VALOR_ARRECADADO'], axis=1)
        row = np.array(row_.iloc[-1]).reshape((1, -1))
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        row_norm = input_scaler.fit_transform(row)
        to_prev = row_norm.reshape((row_norm.shape[0], 1, row_norm.shape[1]))
        prev = model_lstm.predict(to_prev)
        prev = scaler_y.inverse_transform(prev)
        st.write(prev)
        st.write(prev[0][0])
        row_.insert(1, 'VALOR_ARRECADADO', prev[0][0])
        date_df = pd.concat([date_df, row_], axis=0, ignore_index=True)
    return date_df[0][0]


# Previsão de Receitas
with st.form("user_form"):
    st.write("## Previsão de Receitas")
    st.write("### Informe a data para a previsão")
    input_date = st.date_input("Data")
    submit_button = st.form_submit_button(label='Enviar')

if submit_button:
    predicted_revenue = predict_revenue(input_date, df_receitas, model_lstm)
    st.write(f'Previsão de Receitas para {input_date}: R$ {predicted_revenue:.2f}')