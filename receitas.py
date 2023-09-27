import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import streamlit as st
from PIL import Image
import os

# Funções de Carregamento e Pré-processamento de Dados
@st.cache_data()
def load_model_and_data():
    model_lstm = tf.keras.models.load_model('receitas_2013_2022')
    dados_receitas = pd.read_csv('data/dados_receitas.csv', sep=';')
    return model_lstm, dados_receitas

def beteween_dates(input_date, df_date):
    date_01 = pd.to_datetime(input_date)
    date_02 = dt.date.fromordinal(df_date)
    years = relativedelta(date_01, date_02).years
    months = relativedelta(date_01, date_02).months
    if years > 0:
        return (years * 12) + months
    else:
        return months

# Funções de Previsão
def prediction(model, X, scaler_y):
    y_pred = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred)
    return y_pred

def predict_revenue(input_date, df_receitas, model_lstm, input_scaler, scaler_y):
    date_df = df_receitas.copy()
    st.write(f"Previsão de Receitas para {beteween_dates(input_date, date_df['ANO_MES'].iloc[-1])} meses à frente do último arquivo da base de dados utilizada para treino do modelo.")
    for i in range(beteween_dates(input_date, date_df['ANO_MES'].iloc[-1])):
        # Crie uma nova linha de dados vazia
        row = pd.DataFrame(columns=date_df.columns)

        # Calcule as médias e valores de atraso
        row.loc[0, 'SMA(12)'] = date_df['VALOR_ARRECADADO'].iloc[-12:].mean()
        row.loc[0, 'SMA(6)'] = date_df['VALOR_ARRECADADO'].iloc[-6:].mean()
        row.loc[0, 'SMA(3)'] = date_df['VALOR_ARRECADADO'].iloc[-3:].mean()
        row.loc[0, 'SMA(2)'] = date_df['VALOR_ARRECADADO'].iloc[-2:].mean()
        row.loc[0, 'lag(12)'] = date_df['VALOR_ARRECADADO'].iloc[-12]
        row.loc[0, 'lag(6)'] = date_df['VALOR_ARRECADADO'].iloc[-6]
        row.loc[0, 'lag(4)'] = date_df['VALOR_ARRECADADO'].iloc[-4]
        row.loc[0, 'lag(3)'] = date_df['VALOR_ARRECADADO'].iloc[-3]
        row.loc[0, 'lag(2)'] = date_df['VALOR_ARRECADADO'].iloc[-2]
        row.loc[0, 'lag(1)'] = date_df['VALOR_ARRECADADO'].iloc[-1]

        # Incremente a data
        row.loc[0, 'ANO_MES'] = date_df['ANO_MES'].iloc[-1]+1
        
        # Excluindo a coluna de valor arrecadado
        row = row.drop(['VALOR_ARRECADADO'], axis=1)
        
        # Transforme a linha em um array e normalize
        row = np.array(row.iloc[-1]).reshape(1, -1)
        row_norm = input_scaler.transform(row)

        # Preveja usando o modelo LSTM
        to_prev = row_norm.reshape((row_norm.shape[0], 1, row_norm.shape[1]))
        prev = model_lstm.predict(to_prev)
        prev = scaler_y.inverse_transform(prev)

        # Crie um DataFrame com a previsão e adicione ao DataFrame principal
        row_ = pd.DataFrame(row, columns = ['ANO_MES', 'SMA(12)', 'SMA(6)', 'SMA(3)', 'SMA(2)', 'lag(12)', 'lag(6)', 'lag(4)', 'lag(3)', 'lag(2)', 'lag(1)'])
        row_.loc[0, 'VALOR_ARRECADADO'] = prev[0]
        date_df = pd.concat([date_df, row_], ignore_index=True)
    return date_df.at[date_df.index[-1], 'VALOR_ARRECADADO']

# Funções de Visualização
def display_images():
    st.title('Acurácia do Modelo de Previsão de Receitas Municipais')
    image_path_01 = 'src/static/images/figura[4].png'
    image_path_02 = 'src/static/images/figura[5].png'

    # Verifique se as imagens existem
    if os.path.exists(image_path_01) and os.path.exists(image_path_02):
        image_01 = Image.open(image_path_01)
        st.image(image_01, caption='Figura 1: Gráfico de previsão de receitas municipais, pesquisa e treinamento do modelo.\nFonte: Elaboração própria.', width=800)

        image_02 = Image.open(image_path_02)
        st.image(image_02, caption='Figura 2: Distribuição de erros.\nFonte: Elaboração própria.', width=800)
    else:
        st.error('Imagens não encontradas. Verifique os caminhos das imagens.')

def remove_outliers(dados_receitas):
    Q1 = dados_receitas['VALOR_ARRECADADO'].quantile(0.25)
    Q3 = dados_receitas['VALOR_ARRECADADO'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df_sem_outliers = dados_receitas[(dados_receitas['VALOR_ARRECADADO'] >= limite_inferior) & (dados_receitas['VALOR_ARRECADADO'] <= limite_superior)]
    return df_sem_outliers

def preprocess_data(df_sem_outliers):
    df_sem_outliers['DATA'] = pd.to_datetime(df_sem_outliers['DATA'])
    df_sem_outliers['ANO_MES'] = df_sem_outliers['DATA'].dt.strftime('%Y-%m')
    df_sem_outliers['ANO'] = df_sem_outliers['DATA'].dt.year
    df_sem_outliers = df_sem_outliers.groupby(['ANO_MES', 'ANO'])['VALOR_ARRECADADO'].sum().reset_index()
    df_sem_outliers['SMA(12)'] = df_sem_outliers['VALOR_ARRECADADO'].rolling(window=12).mean()
    df_sem_outliers['SMA(6)'] = df_sem_outliers['VALOR_ARRECADADO'].rolling(window=6).mean()
    df_sem_outliers['SMA(3)'] = df_sem_outliers['VALOR_ARRECADADO'].rolling(window=3).mean()
    df_sem_outliers['SMA(2)'] = df_sem_outliers['VALOR_ARRECADADO'].rolling(window=2).mean()
    df_sem_outliers['lag(12)'] = df_sem_outliers['VALOR_ARRECADADO'].shift(12)
    df_sem_outliers['lag(6)'] = df_sem_outliers['VALOR_ARRECADADO'].shift(6)
    df_sem_outliers['lag(4)'] = df_sem_outliers['VALOR_ARRECADADO'].shift(4)
    df_sem_outliers['lag(3)'] = df_sem_outliers['VALOR_ARRECADADO'].shift(3)
    df_sem_outliers['lag(2)'] = df_sem_outliers['VALOR_ARRECADADO'].shift(2)
    df_sem_outliers['lag(1)'] = df_sem_outliers['VALOR_ARRECADADO'].shift(1)
    df_sem_outliers.dropna(inplace=True)
    df_sem_outliers.drop(columns=['ANO'], inplace=True)
    df_sem_outliers['ANO_MES'] = pd.to_datetime(df_sem_outliers['ANO_MES'])
    df_sem_outliers['ANO_MES'] = df_sem_outliers['ANO_MES'].map(dt.datetime.toordinal)
    df_sem_outliers.reset_index(drop=True, inplace=True)
    return df_sem_outliers

def scale_data(df_sem_outliers):
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X = df_sem_outliers.drop(['VALOR_ARRECADADO'], axis=1)
    y = df_sem_outliers.loc[:, ['VALOR_ARRECADADO']]

    st.write(X)
    st.write(y)

    input_scaler = scaler_x.fit(X)
    output_scaler = scaler_y.fit(y)

    y_norm = output_scaler.transform(y)
    X_norm = input_scaler.transform(X)

    st.write(X_norm)
    st.write(y_norm)

    X = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
    y = y_norm.reshape((y_norm.shape[0], 1))
    st.write(X)
    st.write(y)

    y = scaler_y.inverse_transform(y)
    return X, y, input_scaler, output_scaler, scaler_y

def calculate_mean_error(y, y_pred):
    real = y.flatten()
    previsto = y_pred.flatten()
    tabela = pd.DataFrame([real, previsto]).T
    tabela = tabela.rename(columns={0: 'Real', 1: 'Previsto'})
    tabela['Diferenca'] = 1 - (tabela['Real'] / tabela['Previsto'])
    media_tabela = (tabela['Diferenca'].mean() * 100)
    return media_tabela

def plot_future(prediction, y):
    plt.figure(figsize=(10,6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y), label='Dados reais')
    plt.plot(np.arange(range_future), np.array(prediction),label='Predição')
    plt.legend(loc='upper left')
    plt.xlabel('Período')
    plt.ylabel('Valor Arrecadado')
    plt.title('Predição de Receitas - LSTM')
    plt.savefig('src/static/images/figura[6].png')
    image_path_03 = 'src/static/images/figura[6].png'
    # Verifique se as imagens existem
    if os.path.exists(image_path_03):
        image_03 = Image.open(image_path_03)
        st.image(image_03, caption='Figura 3: Gráfico de previsão de receitas municipais, dados e modelo atuais.\nFonte: Elaboração própria.', width=800)
    else:
        st.error('Imagens não encontradas. Verifique os caminhos das imagens.')

# Funções Principais
def main():
    st.title('Previsão de Receitas Municipais')
    st.markdown('Este aplicativo tem como objetivo prever as receitas municipais do Município de Cajazeiras/PB, com base em dados históricos de arrecadação.'
             'Para isso, foi utilizado um modelo de redes neurais recorrentes (RNN) do tipo LSTM (Long Short-Term Memory).')
    display_images()

    model_lstm, dados_receitas = load_model_and_data()

    # ... (código principal omitido para economizar espaço)
    df_sem_outliers = remove_outliers(dados_receitas)
    df_sem_outliers = preprocess_data(dados_receitas)
    X, y, input_scaler, output_scaler, scaler_y = scale_data(df_sem_outliers)
    y_pred = prediction(model_lstm, X, scaler_y)
    plot_future(y_pred, y)
    media_tabela = calculate_mean_error(y, y_pred)
    st.title(f'Erro médio percentual: {media_tabela:.2f}%')

    # Previsão de Receitas
    with st.form("user_form"):
        st.write("## Previsão de Receitas")
        st.write("### Informe a data para a previsão")
        input_date = st.date_input("Data")
        submit_button = st.form_submit_button(label='Enviar')

    if submit_button:
        predicted = predict_revenue(input_date, df_sem_outliers, model_lstm, input_scaler, scaler_y)
        st.write(f'Previsão de Receitas para {input_date}: R$ {predicted:.2f}')
        st.write(f'Erro percentual: {((predicted - y[-1][0]) / y[-1][0]) * 100:.2f}%')

if __name__ == "__main__":
    main()