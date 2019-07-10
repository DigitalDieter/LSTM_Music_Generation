from data_utils.parse_files import convert_folder_to_wav
from data_utils.parse_files import convert_wav_files_to_nptensor
import config.nn_config as nn_config


CONFIG = nn_config.get_neural_net_configuration()

INPUT_DIRECTORY = CONFIG['dataset_directory']

OUTPUT_FILENAME = CONFIG['model_file']

FREQ = CONFIG['sampling_frequency'] # sample frequency in Hz
# The author defines "clip_len", "block_size" and "max_seq_len"

# length of clips for training. Defined in seconds
CLIP_LEN = 10

# block sizes used for training - this defines the size of our input state
BLOCK_SIZE = FREQ / 4

# Used later for zero-padding song sequences
MAX_SEQ_LEN = int(round((FREQ * CLIP_LEN) / BLOCK_SIZE))

# Step 1 - convert MP3s to WAVs
NEW_DIRECTORY = convert_folder_to_wav(INPUT_DIRECTORY, FREQ)

# Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
convert_wav_files_to_nptensor(NEW_DIRECTORY, BLOCK_SIZE, MAX_SEQ_LEN, OUTPUT_FILENAME)
