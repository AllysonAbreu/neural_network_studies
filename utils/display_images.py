import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Funções de Visualização
def display_images():
    st.title('Acurácia do Modelo de Previsão de Receitas Municipais')
    image_path_01 = 'src/static/images/receitas/figura[3].png'
    image_path_02 = 'src/static/images/receitas/figura[5].png'

    # Verifique se as imagens existem
    if os.path.exists(image_path_01) and os.path.exists(image_path_02):
        image_01 = Image.open(image_path_01)
        st.image(image_01, caption='Figura 1: Gráfico de previsão de receitas municipais, pesquisa e treinamento do modelo.\nFonte: Elaboração própria.', width=800)

        image_02 = Image.open(image_path_02)
        st.image(image_02, caption='Figura 2: Distribuição de erros.\nFonte: Elaboração própria.', width=800)
    else:
        st.error('Imagens não encontradas. Verifique os caminhos das imagens.')

def plot_future(prediction, y):
    plt.figure(figsize=(10,6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y), label='Dados reais')
    plt.plot(np.arange(range_future), np.array(prediction),label='Predição')
    plt.legend(loc='upper left')
    plt.xlabel('Período')
    plt.ylabel('Valor Arrecadado')
    plt.title('Predição de Receitas - LSTM')
    plt.savefig('src/static/images/receitas/figura[6].png')
    image_path_03 = 'src/static/images/receitas/figura[6].png'
    # Verifique se as imagens existem
    if os.path.exists(image_path_03):
        image_03 = Image.open(image_path_03)
        st.image(image_03, caption='Figura 3: Gráfico de previsão de receitas municipais, dados e modelo atuais.\nFonte: Elaboração própria.', width=800)
    else:
        st.error('Imagens não encontradas. Verifique os caminhos das imagens.')