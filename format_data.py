import os
import pickle
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

sequence_length = 1000

def count_lengths(dirname):
    lengths = set()
    min_len = None
    for filename in os.listdir(dirname):
        freq, data = wavfile.read(dirname + "/" + filename)
        data = np.array(data, dtype=float)
        lengths.add(data.shape[0])
        if not min_len or data.shape[0] < min_len:
            min_len = data.shape[0]
    return min_len

def normalize_files(min_length):
    files = os.listdir("clips_wav")
    sequences = np.zeros((len(files), sequence_length))
    id_map = []

    for i, filename in enumerate(files):
        song_id = int(filename.split('.')[0])
        id_map.append(song_id)

        freq, data = wavfile.read("clips_wav/" + filename)
        data = np.array(data, dtype=float)[:min_length]
        data = signal.resample(data, sequence_length)
        sequences[i, :] = data

    with open("wav_data_" + str(sequence_length) + ".pkl", 'wb') as f:
        pickle.dump((sequences, id_map), f)


if __name__ == "__main__":
    min_len = count_lengths("clips_wav")
    normalize_files(min_len)
    # data = pickle.load(open("wav_data_1000.pkl", "rb"))
