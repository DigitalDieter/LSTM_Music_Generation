from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
import config.nn_config as nn_config


CONFIG = nn_config.get_neural_net_configuration()

# linear
ACTIVATION = CONFIG['activation']
STATEFUL = CONFIG['stateful']

FREQ_SPACE_DIMS = CONFIG['num_frequency_dimensions']
NUM_HIDDEN_DIMENSIONS = CONFIG['hidden_dimension_size']
NUM_RECURRENT_UNITS = CONFIG['num_recurrent_units']

def create_lstm_network(FREQ_SPACE_DIMS, NUM_HIDDEN_DIMENSIONS, NUM_RECURRENT_UNITS):
    model = Sequential()
    # Layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(NUM_HIDDEN_DIMENSIONS, activation=ACTIVATION),
                              input_shape=(None, FREQ_SPACE_DIMS)))

    for cur_unit in range(NUM_RECURRENT_UNITS):
        model.add(LSTM(units=NUM_HIDDEN_DIMENSIONS, return_sequences=True, stateful=STATEFUL))
    # Layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(input_dim=NUM_HIDDEN_DIMENSIONS, output_dim=FREQ_SPACE_DIMS,
                                    activation=ACTIVATION)))

    # Done building the model.Now, configure it for the learning process
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


def create_gru_network(num_frequency_dimensions, NUM_HIDDEN_DIMENSIONS):
    model = Sequential()
    # Layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(NUM_HIDDEN_DIMENSIONS), input_shape=(None, num_frequency_dimensions)))
    model.add(GRU(units=NUM_HIDDEN_DIMENSIONS, return_sequences=True))
    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
