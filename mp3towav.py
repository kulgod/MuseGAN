from pydub import AudioSegment

def main():
    for i in range(2, 3):
        filename = 'clips_45seconds/' + str(i) + '.mp3'
        mp3_to_wav = AudioSegment.from_mp3(filename)
        mp3_to_wav.export(open('clips_wav/' + str(i) + '.wav', 'wb'), format='wav')


if __name__ == "__main__":
    main()
