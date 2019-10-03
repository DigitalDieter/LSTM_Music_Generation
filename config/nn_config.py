import os

def get_neural_net_configuration():
    nn_params = {}
    nn_params['current_iteration'] = 0
    nn_params['num_frequency_dimensions'] = 22050
    nn_params['sampling_frequency'] = 44100
    nn_params['num_recurrent_units'] = 1
    nn_params['hidden_dimension_size'] = 1024

    # The weights filename for saving/loading trained models
    nn_params['stateful'] = False
    nn_params['return_sequences'] = True
    nn_params['loss_function'] = 'mean_squared_error'
    nn_params['optimizer'] = 'rmsprop'
    nn_params['metrics'] = ['accuracy']

    if not os.path.exists('losses'):
        os.makedirs("losses")
        nn_params['lossfile_basename'] = 'losses/LSTM_loss_iter-'
    else:
        nn_params['lossfile_basename'] = 'losses/LSTM_loss_iter-'

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
