import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Reshape, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import save_model, load_model
import math


def get_fully_connected_model(input_shape, data_name):
    num_bits = input_shape[0]
    model_name = f'{num_bits}_FC_{data_name}_flat_d512_drop03_d256_drop02_d128_d1'
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(.3, noise_shape=None, seed=None))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(.2, noise_shape=None, seed=None))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return (model, model_name)


def get_convolutional_model(input_shape, data_name):
    print(input_shape)
    num_bits = input_shape[0]
    dimension = int(math.sqrt(num_bits))
    model_name = f'{num_bits}_CNN_{data_name}_resh_conv1d128-16_drop03_pool1d_conv1d64-8_drop02_pool1d_flat_d1'
    model = Sequential()
    model.add(Reshape((dimension, dimension, 1), input_shape=input_shape))
    model.add(Conv2D(128, 3, activation='relu',
                     input_shape=(dimension, dimension)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, activation='relu'))
    '''
    CURRENTLY BEST!
    model.add(Conv2D(128, 3, activation='relu',
                     input_shape=(dimension, dimension)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Dropout(0.1))
    '''
    """
    model.add(Conv2D(
        filters=64, kernel_size=3, activation='relu', input_shape=[dimension, dimension]))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(Dropout(0.1, noise_shape=None, seed=None)) """
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return (model, model_name)


def get_lstm_model(input_shape, data_name):
    num_bits = input_shape[0]
    model_name = f'{num_bits}_LSTM_{data_name}_resh32x16_LSTM256_drop03_LSTM128_drop02_d64_d1'
    model = Sequential()
    model.add(Reshape((10, int(num_bits / 10)), input_shape=input_shape))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1, noise_shape=None, seed=None))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return (model, model_name)


def get_lstm_model_stateful(input_shape, data_name):
    '''
    Currently not working
    '''
    num_bits = input_shape[0]
    model_name = f'{num_bits}_LSTM_{data_name}_resh32x16_LSTM256_drop03_LSTM128_drop02_d64_d1'
    model = Sequential()
    model.add(Reshape((4, int(num_bits / 4)), input_shape=input_shape))
    model.add(LSTM(256, activation='relu', return_sequences=True,
                   stateful=True, batch_input_shape=(200, 4, 64)))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(LSTM(128, activation='relu', stateful=True))
    model.add(Dropout(0.1, noise_shape=None, seed=None))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return (model, model_name)


def get_defined_models(input_shape, data_name):
    return [
        get_fully_connected_model(input_shape, data_name),
        get_convolutional_model(input_shape, data_name),
        get_lstm_model(input_shape, data_name)
    ]
