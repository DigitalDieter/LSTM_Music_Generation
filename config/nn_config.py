import os

def get_neural_net_configuration():
    nn_params = {}
    nn_params['sampling_frequency'] = 44100

    nn_params['num_recurrent_units'] = 1
    # Number of hidden dimensions.
    # For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
    nn_params['hidden_dimension_size'] = 1024
    # The weights filename for saving/loading trained models
    nn_params['stateful'] = False
    if not os.path.exists('weights'):
        os.makedirs("weights")
        nn_params['model_basename'] = 'weights/np_weights_iter-'
    else:
        nn_params['model_basename'] = 'weights/np_weights_iter-'

    # The model filename for the training data
    nn_params['model_file'] = './datasets/np_array'
    # The dataset directory
    nn_params['dataset_directory'] = './datasets/training_data/'

    nn_params['activation'] = 'linear'
    return nn_params
