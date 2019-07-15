import numpy as np
from keras import backend as K


# Extrapolates from a given seed sequence

def generate_from_seed(model, seed, sequence_length, data_variance, data_mean):
    seedSeq = seed.copy()
    output = []

    for it in range(sequence_length):
        seedSeqNew = model.predict(seedSeq)  # Step 1. Generate X_n + 1
        #print("seedSeqNew",seedSeqNew.shape)
        # Step 2. Append it to the sequence
        if it == 0:
            for i in range(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())
        else:
            output.append(seedSeqNew[0][seedSeqNew.shape[1] - 1].copy())
        newSeq = seedSeqNew[0][seedSeqNew.shape[1] - 1]
        newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))

        seedSeq = np.concatenate((seedSeq[:,1:,:], newSeq), axis=1)


    return output

