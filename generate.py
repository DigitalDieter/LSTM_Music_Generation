from __future__ import absolute_import
from __future__ import print_function
import glob
import os
import numpy as np
import inquirer
import nn_utils.network_utils as network_utils
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
from gen_utils.filename_generator import saveSong
from data_utils.parse_files import save_generated_example
import config.nn_config as nn_config



#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
CONFIG = nn_config.get_neural_net_configuration()

# Select  weights file for generating song
MODEL_WEIGTHS = [
    inquirer.List('size',
                  message="Please choose saved weights file for generating the song",
                  choices=glob.glob('weights/*.h5')
                  ),
]

CHOOSE_MODEL = inquirer.prompt(MODEL_WEIGTHS)

SAMPLE_FREQUENCY = CONFIG['sampling_frequency']

INPUTFILE = CONFIG['model_file']
MODEL_BASENAME = CONFIG['model_basename']
CUR_ITER = 10

MODEL_FILENAME = CHOOSE_MODEL["size"]

print("sample_frequency:", SAMPLE_FREQUENCY)
print("INPUTFILE:", INPUTFILE)
print("MODEL_BASENAME:", MODEL_BASENAME)
print("cur_iter:", CUR_ITER)
print("model_filename:", MODEL_FILENAME)


OUTPUT_FILENAME = saveSong()


# Load up the training data
print('Loading training data')
# X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
# y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
# X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
# X_var is a matrix of size (num_frequency_dims,)
# containing the variance for each frequency dimension
X_TRAIN = np.load(INPUTFILE + '_x.npy')
Y_TRAIN = np.load(INPUTFILE + '_y.npy')
X_MEAN = np.load(INPUTFILE + '_mean.npy')
X_VAR = np.load(INPUTFILE + '_var.npy')
print('Finished loading training data')

# Figure out how many frequencies we have in the data

FREQ_SPACE_DIMS = X_TRAIN.shape[2]
HIDDEN_DIMS = CONFIG['hidden_dimension_size']

# Creates a lstm network
MODEL = network_utils.create_lstm_network(num_frequency_dimensions=FREQ_SPACE_DIMS,
                                          NUM_HIDDEN_DIMENSIONS=HIDDEN_DIMS, NUM_RECURRENT_UNITS=1)


# Load existing weights if available
if os.path.exists(MODEL_FILENAME):
    MODEL.load_weights(MODEL_FILENAME)
else:
    print('Model filename ' + MODEL_FILENAME + ' could not be found!')

def show_values():
    for k, v in CONFIG.items():
        print(k + ': ' + str(v))

show_values()
print('Starting generation!')

SEED_LEN = 1
SEED_SEQ = seed_generator.generate_copy_seed_sequence(seed_length=SEED_LEN,
                                                      training_data=X_TRAIN)

# Defines final song length. Total song length in samples = max_seq_len * example_len
MAX_SEQ_LEN = 10

OUTPUT = sequence_generator.generate_from_seed(model=MODEL, seed=SEED_SEQ,
                                               data_variance=X_VAR, data_mean=X_MEAN, sequence_length=MAX_SEQ_LEN)

print('Finished generation!')

# Save the generated sequence to a WAV file
save_generated_example(str(OUTPUT_FILENAME), OUTPUT, sample_frequency=SAMPLE_FREQUENCY)
