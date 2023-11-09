import pandas as pd
import tensorflow as tf
import streamlit as st
from utils import *

# Funções de Carregamento e Pré-processamento de Dados
@st.cache_data()
def load_model_and_data():
    model_lstm = tf.keras.models.load_model('notebooks/v2/receitas_2013_2022')
    dados_receitas = pd.read_csv('data/dados_receitas_com_outliers.csv', sep=';')
    return model_lstm, dados_receitas

# Funções Principais
def main():
    st.title('Previsão de Receitas Municipais')
    st.markdown('Este aplicativo tem como objetivo prever as receitas municipais do Município de Cajazeiras/PB, com base em dados históricos de arrecadação.'
             'Para isso, foi utilizado um modelo de redes neurais recorrentes (RNN) do tipo LSTM (Long Short-Term Memory).')
    display_images.display_images()

    model_lstm, dados_receitas = load_model_and_data()
    X, y, input_scaler, output_scaler, scaler_y = data_enginnering.scale_data(dados_receitas)
    y_pred = predict_functions.prediction(model_lstm, X, scaler_y)
    display_images.plot_future(y_pred, y)
    media_tabela = data_enginnering.calculate_mean_error(y, y_pred)
    st.title(f'Erro médio percentual: {media_tabela:.2f}%')

    # Previsão de Receitas
    with st.form("user_form"):
        st.write("## Previsão de Receitas")
        st.write("### Informe a data para a previsão")
        input_date = st.date_input("Data")
        submit_button = st.form_submit_button(label='Enviar')

    if submit_button:
        predicted = predict_functions.predict_revenue(input_date, dados_receitas, model_lstm, input_scaler, scaler_y)
        valor_com_erro_pecentual = predicted * media_tabela
        st.write(f'Previsão de Receitas para {input_date}: R$ {valor_com_erro_pecentual:.2f}')

if __name__ == "__main__":
    main()