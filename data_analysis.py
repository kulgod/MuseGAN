from scipy.io import wavfile
import os
import numpy as np

def main():
    lengths = set()
    count = 0
    min = None
    for filename in os.listdir("clips_wav"):
        freq, data = wavfile.read("clips_wav/" + filename)
        data = np.array(data, dtype=float)
        lengths.add(data.shape[0])
        count += 1
        if not min or data.shape[0] < min:
            min = data.shape[0]
    print(count, min, len(lengths))

if __name__ == "__main__":
    main()
