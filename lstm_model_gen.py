from io import BytesIO
import base64
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from plot_generator import get_data


# Declare sequence_length as a global variable
sequence_length = 10

def create_sequences(data, sequence_length, fields=['Open', 'High', 'Low', 'Close']):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i + sequence_length][fields]
        target = data.iloc[i + sequence_length]['Close']

        sequences.append(sequence)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=1024, step=64),
                           input_shape=(sequence_length, 4)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def find_best_model(X_train, y_train, X_test, y_test,max_trials = 2,epochs = 10):
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,
        directory='tuner_logs',
        project_name='my_lstm_tuner'
    )

    tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.compile(optimizer='adam', loss='mean_squared_error')
    #best_model.predict()
    # Specify the number of epochs and batch size
    #epochs = 50  # Adjust this value as needed
    #batch_size = 32  # Adjust this value as needed

    # Train the best model on the training data with validation data
    #history = best_model.fit(
    #  X_train, y_train,
    #  epochs=epochs,
    #  batch_size=batch_size,
    #  validation_data=(X_test, y_test)  # Use the validation data for monitoring
    #)


    # Evaluate the best model on the test data
    test_loss = best_model.evaluate(X_test, y_test)
    print("---------------------------- test loss ----------------------")
    print(test_loss)
    print("---------------------------- end test loss ----------------------")
    return best_model
    
def train_best_model(model,X_train, y_train, X_test, y_test,epochs=10,batch_size=32 ):
  #epochs = 50  # Adjust this value as needed
  #batch_size = 32  # Adjust this value as needed

  # Train the best model on the training data with validation data
  history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test)  # Use the validation data for monitoring
  )
  return model
  
  
def predict_with_model(model, sequence):
    return model.predict(sequence)[0]

def generate_lstm_plot(currency_pairs, periods):
    data_plot = []
    charts_params = [{'5m':'59d'},{'15m':'59d'},{'1h':'729d'},{'1d':'30000d'}]
    best_model = None
    for c in currency_pairs:
        #print(c)
        for p in periods:
            period = [item[p] for item in charts_params if p in item]
            #print(period[0])
            data = get_data(c,"","",period[0],p)
            num_features = 4
            scaler = MinMaxScaler()
            data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']])

            sequence_length = 10
            X, y = create_sequences(data[['Open', 'High', 'Low', 'Close']], sequence_length, ['Open', 'High', 'Low', 'Close'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            #best_model = None
            #if best_model is None:
            print("-----------------")
            best_model = None
            best_model = find_best_model(X_train, y_train, X_test, y_test,5,15)
            #else:
            #    best_model = train_best_model(best_model,X_train, y_train, X_test, y_test,5,15 )
            recent_data = data[['Open', 'High', 'Low', 'Close']][-sequence_length:].values
            predicted_values = []
            num_steps_to_predict = 15
            for _ in range(num_steps_to_predict):
                predicted_value = predict_with_model(best_model, recent_data.reshape(1, sequence_length, num_features))
                predicted_values.append(predicted_value)
                new_sequence = recent_data.copy()
                new_sequence[:-1] = new_sequence[1:]
                new_sequence[-1] = predicted_value[-1]
                recent_data = new_sequence
                
            predicted_values = np.array([[predicted_values]])
            shape = predicted_values.shape
            zeros_array = np.zeros((shape[0], shape[1], shape[2], 4))
            zeros_array[:, :, :, -shape[-1]:] = predicted_values
            reshaped_zeros_array = zeros_array.reshape(-1, 4)
            predicted_values_inverse = scaler.inverse_transform(reshaped_zeros_array)
            last_column_values = [row[-1] for row in predicted_values_inverse]
            last_column_values = np.array(last_column_values)

            actual_values = data[['Open', 'High', 'Low', 'Close']].values[-50:]
            actual_values = scaler.inverse_transform(actual_values)
            last_column_actual_values = [row[-1] for row in actual_values]
            last_column_actual_values = np.array(last_column_actual_values)
            
            # Plot the historical data and forecasted values
            plt.figure(figsize=(10, 6))
            
            plt.clf()
            plt.plot(range(len(last_column_actual_values)), last_column_actual_values, label='Actual Data')
            plt.plot(range(len(last_column_actual_values), len(last_column_actual_values) + len(last_column_values)), last_column_values, label='Predicted Values', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('Close Price')
            plt.title(f"Actual and Forecasted Close {p} Prices for {c}")
            plt.legend()
            plt.grid(True)   
            # Save the plot to a BytesIO object
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_plot.append({
                'plot_data': img_data,
                'currency_pairs': f"{c} LSTM Forecast]",
                'periods': p
            })
            
    return data_plot