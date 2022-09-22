from tensorflow import keras
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler


def load_doc(path):
    dataset = pd.read_csv(path, delimiter =',', parse_dates=['timestamp'], index_col ='timestamp')
    dataset.head()
    #date = pd.to_datetime(dataset.pop('DATE'), format='%d.%m.%Y')
    return dataset

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range (len(X) - time_steps):
        v = X.iloc[i: (i+time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

def predict(days):
    df = pd.read_csv('covid_number.csv', delimiter=',', parse_dates=['DATE'], index_col='DATE')

    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month

    #df_by_month = df.resample('M').sum()
    #sns.pointplot(x=df.index, y='NO_HOSPITALIZED', data=df)
    #plt.show()
    #df_by_day_of_week = df.resample('D').sum()
    #sns.lineplot(data=df_by_day_of_week, x='day_of_week', y = 'NO_HOSPITALIZED')
    #plt.show()

    train_size = int(len(df) * 0.9)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    # print(train.shape, test.shape)

    # f_transformer = RobustScaler()
    hosp_transformer = RobustScaler()

    hosp_transformer = hosp_transformer.fit(train[['NO_HOSPITALIZED']])

    train['NO_HOSPITALIZED'] = hosp_transformer.transform(train[['NO_HOSPITALIZED']])
    test['NO_HOSPITALIZED'] = hosp_transformer.transform(test[['NO_HOSPITALIZED']])

    TIME_STEPS = 30
    X_train, y_train = create_dataset(train, train.NO_HOSPITALIZED, TIME_STEPS)
    X_test, y_test = create_dataset(test, test.NO_HOSPITALIZED, TIME_STEPS)
    #print(X_train == y_train)


    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=128,
                input_shape=(X_train.shape[1], X_train.shape[2])
            )
        )
    )
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train, y_train, epochs=50, batch_size=200, validation_split=0.1, shuffle=False)

    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='validation')
    #plt.legend()
    #plt.show()

    y_pred = model.predict(X_test)
    y_train_inv = hosp_transformer.inverse_transform(y_train.reshape(1, -1))
    y_test_inv = hosp_transformer.inverse_transform(y_test.reshape(1, -1))
    y_pred_inv = hosp_transformer.inverse_transform(y_pred)

    #plt.plot(y_test_inv.flatten(), marker='.', label='true')
    #plt.plot(y_pred_inv.flatten(), 'r', label='predicted')
    #plt.legend()
    #plt.show()

    return y_pred_inv



def predict_from_model(day):
    df = pd.read_csv('covid_number.csv', delimiter=',', parse_dates=['DATE'], index_col='DATE')

    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month

    df_by_month = df.resample('M').sum()
    df_by_day_of_week = df.resample('D').sum()


    train_size = int(len(df) * 0.9)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    hosp_transformer = RobustScaler()

    hosp_transformer = hosp_transformer.fit(train[['NO_HOSPITALIZED']])

    train['NO_HOSPITALIZED'] = hosp_transformer.transform(train[['NO_HOSPITALIZED']])
    test['NO_HOSPITALIZED'] = hosp_transformer.transform(test[['NO_HOSPITALIZED']])

    TIME_STEPS = 30  # moguc unos
    X_train, y_train = create_dataset(train, train.NO_HOSPITALIZED, TIME_STEPS)
    X_test, y_test = create_dataset(test, test.NO_HOSPITALIZED, TIME_STEPS)

    model = tf.keras.models.load_model('goodModel.h5', custom_objects=None, compile=True, options=None)

    y_pred = model.predict(X_test)
    y_test_inv = hosp_transformer.inverse_transform(y_test.reshape(1, -1))
    y_pred_inv = hosp_transformer.inverse_transform(y_pred)

    #plt.plot(y_pred_inv.flatten(), 'r', label='predicted')
    #plt.legend()
    #plt.show()

    return y_pred_inv