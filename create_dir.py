import os

# create file structure for training data
if not os.path.exists('datasets/training_data'):
    os.makedirs("datasets/training_data/")
    os.makedirs("datasets/training_data/wave")
    os.makedirs("datasets/training_data/tmp")

else:
    print("folder already existing")