from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed
from tensorflow.keras.layers import GRU, LSTM
import config.nn_config as nn_config


CONFIG = nn_config.get_neural_net_configuration()

ACTIVATION = CONFIG['activation'] # linear
STATEFUL = CONFIG['stateful'] # False
SEQUENCE = CONFIG['return_sequences'] # True
FREQ_SPACE_DIMS = CONFIG['num_frequency_dimensions'] # 22050
NUM_HIDDEN_DIMENSIONS = CONFIG['hidden_dimension_size'] # 1024
NUM_RECURRENT_UNITS = CONFIG['num_recurrent_units'] # 1
LOSS = CONFIG['loss_function'] # mean_squared_error
OPTIMIZER = CONFIG['optimizer'] # rmsprop
METRICS = CONFIG['metrics'] # ['accuracy']

batch_size = 32


def create_lstm_network(FREQ_SPACE_DIMS, NUM_HIDDEN_DIMENSIONS, NUM_RECURRENT_UNITS):
    model = Sequential()
    # Layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(NUM_HIDDEN_DIMENSIONS, activation=ACTIVATION),
                              input_shape=(None, FREQ_SPACE_DIMS)))

    for cur_unit in range(NUM_RECURRENT_UNITS):
        model.add(LSTM(units=NUM_HIDDEN_DIMENSIONS, return_sequences=SEQUENCE, stateful=STATEFUL))
    # Layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(units=FREQ_SPACE_DIMS)))

    # Done building the model.Now, configure it for the learning process
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model


def create_gru_network(num_frequency_dimensions, NUM_HIDDEN_DIMENSIONS):
    model = Sequential()
    # Layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(NUM_HIDDEN_DIMENSIONS), input_shape=(None, FREQ_SPACE_DIMS)))
    model.add(GRU(units=NUM_HIDDEN_DIMENSIONS, return_sequences=SEQUENCE))
    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model
