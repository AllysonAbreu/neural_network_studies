import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    train_size = int(len(df) * 0.75)
    train_dataset, test_dataset = df.iloc[:train_size], df.iloc[train_size:]

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train = train_dataset.drop(['valor_receita'], axis=1)
    y_train = train_dataset.loc[:, ['valor_receita']]

    input_scaler = scaler_x.fit(X_train)
    output_scaler = scaler_y.fit(y_train)

    # normalizando os dados
    y_norm = output_scaler.transform(y_train)
    X_norm = input_scaler.transform(X_train)

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