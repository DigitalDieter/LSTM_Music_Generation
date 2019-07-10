from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
import config.nn_config as nn_config


CONFIG = nn_config.get_neural_net_configuration()

# linear
ACTIVATION = CONFIG['activation']
STATEFUL = CONFIG['stateful']

NUM_RECURRENT_UNITS = CONFIG['num_recurrent_units']

# 1024
NUM_HIDDEN_DIMENSIONS = CONFIG['hidden_dimension_size']


def create_lstm_network(num_frequency_dimensions, NUM_HIDDEN_DIMENSIONS, NUM_RECURRENT_UNITS):
    # Sequential is a linear stack of layers
    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(NUM_HIDDEN_DIMENSIONS, activation=ACTIVATION),
                              input_shape=(None, num_frequency_dimensions)))
    #model.add(TimeDistributed(Dense(input_shape=22050, units=1024, input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions)))
    for cur_unit in range(NUM_RECURRENT_UNITS):
        # return_sequences=True implies return the entire output sequence & not just the last output
        model.add(LSTM(units=NUM_HIDDEN_DIMENSIONS, return_sequences=True, stateful=STATEFUL))
    # This layer converts hidden space back to frequency space

    model.add(TimeDistributed(Dense(input_dim=NUM_HIDDEN_DIMENSIONS, output_dim=num_frequency_dimensions,
                                    activation=ACTIVATION)))

    # Done building the model.Now, configure it for the learning process
    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

