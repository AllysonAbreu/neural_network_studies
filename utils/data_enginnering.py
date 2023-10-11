import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_data(df_sem_outliers):
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X = df_sem_outliers.drop(['valor_receita'], axis=1)
    y = df_sem_outliers.loc[:, ['valor_receita']]

    input_scaler = scaler_x.fit(X)
    output_scaler = scaler_y.fit(y)

    y_norm = output_scaler.transform(y)
    X_norm = input_scaler.transform(X)

    X = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
    y = y_norm.reshape((y_norm.shape[0], 1))

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