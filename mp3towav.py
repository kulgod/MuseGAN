import os
from pydub import AudioSegment

directory = 'clips_45seconds/'

def main():
    for i in range(1, 1001):
        filename = directory + str(i) + '.mp3'
        mp3_to_wav = AudioSegment.from_mp3(filename)
        mp3_to_wav.export(open('clips_wav/' + str(i) + '.wav', 'wb'), format='wav')

if __name__ == "__main__":
    main()
