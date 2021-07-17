# Import Python libraries for LSTM
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def clear_keras_session():
  tf.keras.backend.clear_session()

# Convert an R dataframe into Pandas dataframe
def rdf_to_pandas(r_df):
	df = pd.DataFrame(data=r_df)
	return df
	
def prepare_data(r_df, column):
  data = rdf_to_pandas(r_df)
  values = data[column].values
  return values.reshape((-1,1))
  
def generate_ts_sequence(values, lags, batch_size):
  sequence = TimeseriesGenerator(values, values, length=lags, batch_size=batch_size)
  return sequence

def fit_lstm(train_data, lags, epochs=300, batch_size=45, neurons=40,
             activation="relu", optimizer="rmsprop", loss="mae"):

  train_values = prepare_data(r_df = train_data, column = "value")             
  train_sequence = generate_ts_sequence(train_values, lags, batch_size)

  # Model specification
  model = Sequential()
  # Single hidden layer #1 feature for univariate model
  model.add(LSTM(neurons, return_sequences=False, activation=activation, input_shape=(lags, 1))) 
  # Second hidden layer
  #model.add(LSTM(round(neurons*(1/3)), activation="relu", return_sequences=False))
  # Output layer
  model.add(Dense(1))
  # Compile model
  model.compile(optimizer=optimizer, loss=loss)
  # Train model
  model.fit_generator(train_sequence, epochs=epochs, verbose=1)
  return model

def lsmt_summary_report(model):
  print(model.summary(), flush=True)

def forecast_lstm(lstm_model, test_data, lags, batch_size):
  
  test_values = prepare_data(r_df = test_data, column = "value")             
  test_sequence = generate_ts_sequence(test_values, lags, batch_size)
  
  forecast = lstm_model.predict_generator(test_sequence)
  #print(forecast)
  return forecast.reshape((-1))
