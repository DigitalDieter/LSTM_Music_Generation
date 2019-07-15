from __future__ import absolute_import
from __future__ import print_function
import argparse
import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import nn_utils.network_utils as network_utils
import config.nn_config as nn_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-n', '--n_iter', type=int, default=10,
                    help='number of iterations')
parser.add_argument('-e', '--n_epochs', type=int, default=5,
                    help='epochs per iterations')
parser.add_argument('-b', '--n_batch', type=int, default=5,
                    help='batchsize per iterations')

args = parser.parse_args()

NUM_ITERS = args.n_iter
EPOCHS_PER_ITER = args.n_epochs
BATCH_SIZE = args.n_batch

CUR_ITER = 0
CONFIG = nn_config.get_neural_net_configuration()
INPUTFILE = CONFIG['model_file']
MODEL_BASENAME = CONFIG['model_basename']
MODEL_FILENAME = MODEL_BASENAME + str(CUR_ITER)

# Load up the training data
print('Loading training data')
X_TRAIN = np.load(INPUTFILE + '_x.npy')
Y_TRAIN = np.load(INPUTFILE + '_y.npy')
print('Finished loading training data')

# Figure out how many frequencies we have in the data
FREQ_SPACE_DIMS = CONFIG['num_frequency_dimensions']
HIDDEN_DIMS = CONFIG['hidden_dimension_size']
NUM_RECURR = 1

# Creates a gru network
MODEL = network_utils.create_gru_network(num_frequency_dimensions=FREQ_SPACE_DIMS, NUM_HIDDEN_DIMENSIONS=HIDDEN_DIMS)

# Larger batch sizes require more memory, but training will be faster
print('Starting training!')
WEIGHTS_PATH = 'weights/GRU_NP_Weights_Iter-' + str(NUM_ITERS)
WEIGHTS_NAME = 'GRU_NP_Weights_Iter-' + str(NUM_ITERS)

LOSS = []

while CUR_ITER < NUM_ITERS:
    print('Iteration: ' + str(CUR_ITER))
    HISTORY = MODEL.fit(X_TRAIN, Y_TRAIN, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_ITER, verbose=1, validation_split=0.0)
    LOSS += HISTORY.history['loss']
    with open('gru_losslist.txt', 'a') as filehandle:
        for listitem in LOSS:
            filehandle.write('%s\n' % listitem)
    CUR_ITER += 1


print('Training complete!')
MODEL.save_weights(WEIGHTS_PATH +".h5")
p1 = plt.plot(range(len(LOSS)), LOSS)
plt.title(WEIGHTS_NAME)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig(str(WEIGHTS_PATH) + ".png")
