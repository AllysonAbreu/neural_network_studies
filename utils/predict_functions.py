import streamlit as st
from . import date_util
import pandas as pd
import numpy as np

# Funções de Previsão
def prediction(model, X, scaler_y):
    y_pred = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred)
    return y_pred

def predict_revenue(input_date, df_receitas, model_lstm, input_scaler, scaler_y):
    date_df = df_receitas.copy()
    st.write(f"Previsão de Receitas para {date_util.beteween_dates(input_date, date_df['ano_mes_ordinal'].iloc[-1])} meses à frente do último arquivo da base de dados utilizada para treino do modelo.")
    for i in range(date_util.beteween_dates(input_date, date_df['ano_mes_ordinal'].iloc[-1])):
        # Crie uma nova linha de dados vazia
        row = pd.DataFrame(columns=date_df.columns)

        # Calcule as médias e valores de atraso
        row.loc[0, 'SMA(12)'] = date_df['valor_receita'].iloc[-12:].mean()
        row.loc[0, 'SMA(6)'] = date_df['valor_receita'].iloc[-6:].mean()
        row.loc[0, 'SMA(3)'] = date_df['valor_receita'].iloc[-3:].mean()
        row.loc[0, 'SMA(2)'] = date_df['valor_receita'].iloc[-2:].mean()
        row.loc[0, 'lag(12)'] = date_df['valor_receita'].iloc[-12]
        row.loc[0, 'lag(6)'] = date_df['valor_receita'].iloc[-6]
        row.loc[0, 'lag(4)'] = date_df['valor_receita'].iloc[-4]
        row.loc[0, 'lag(3)'] = date_df['valor_receita'].iloc[-3]
        row.loc[0, 'lag(2)'] = date_df['valor_receita'].iloc[-2]
        row.loc[0, 'lag(1)'] = date_df['valor_receita'].iloc[-1]
        row.loc[0, 'populacao'] = date_df['populacao'].iloc[-1]
        row.loc[0, 'variacao_anual'] = date_df['variacao_anual'].iloc[-1]
        row.loc[0, 'aceleracao_variacao_anual'] = date_df['aceleracao_variacao_anual'].iloc[-1]
        row.loc[0, 'valor_pib'] = date_df['valor_pib'].iloc[-1]


        # Incremente a data
        row.loc[0, 'ano_mes_ordinal'] = date_df['ano_mes_ordinal'].iloc[-1]+1
        
        # Excluindo a coluna de valor arrecadado
        row = row.drop(['valor_receita'], axis=1)
        
        # Transforme a linha em um array e normalize
        row = np.array(row.iloc[-1]).reshape(1, -1)
        row_norm = input_scaler.transform(row)

        # Preveja usando o modelo LSTM
        to_prev = row_norm.reshape((row_norm.shape[0], 1, row_norm.shape[1]))
        prev = model_lstm.predict(to_prev)
        prev = scaler_y.inverse_transform(prev)

        # Crie um DataFrame com a previsão e adicione ao DataFrame principal
        row_ = pd.DataFrame(row, columns = ['ano_mes_ordinal', 'SMA(12)', 'SMA(6)', 'SMA(3)', 'SMA(2)', 'lag(12)', 'lag(6)', 'lag(4)', 'lag(3)', 'lag(2)', 'lag(1)', 'populacao', 'variacao_anual', 'aceleracao_variacao_anual', 'valor_pib'])
        row_.loc[0, 'valor_receita'] = prev[0]
        date_df = pd.concat([date_df, row_], ignore_index=True)
    return date_df.at[date_df.index[-1], 'valor_receita']