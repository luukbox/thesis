import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Reshape, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import save_model, load_model


def get_fully_connected_model(input_shape, data_name):
    num_bits = input_shape[0]
    model_name = f'{num_bits}_FC_{data_name}_flat_d512_drop03_d256_drop02_d128_d1'
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(.2, noise_shape=None, seed=None))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(.2, noise_shape=None, seed=None))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(.1, noise_shape=None, seed=None))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return (model, model_name)


def get_convolutional_model(input_shape, data_name):
    num_bits = input_shape[0]
    model_name = f'{num_bits}_CNN_{data_name}_resh_conv1d128-16_drop03_pool1d_conv1d64-8_drop02_pool1d_flat_d1'
    model = Sequential()
    # reshapes 1D row vector to 2D column vector
    model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
    model.add(Conv1D(
        filters=128, kernel_size=4, activation='relu'))
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return (model, model_name)


def create_lstm_model(input_shape, data_name):
    num_bits = input_shape[0]
    model_name = f'{num_bits}_LSTM_{data_name}_resh32x16_LSTM256_drop03_LSTM128_drop02_d64_d1'
    model = Sequential()
    model.add(Reshape((16, 8), input_shape=input_shape))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return (model, model_name)


def get_defined_models(input_shape, data_name):
    return [
        get_fully_connected_model(input_shape, data_name),
        get_convolutional_model(input_shape, data_name),
        create_lstm_model(input_shape, data_name)
    ]
