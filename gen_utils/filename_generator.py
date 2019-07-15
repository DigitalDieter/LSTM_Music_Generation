import glob
import os
import re


def saveSong():
    currentSong = glob.glob("gen_songs/generated_song*.wav")
    numList = [0]
    for song in currentSong:
        i = os.path.splitext(song)[0]
        try:
            num = re.findall('[0-9]+$', i)[0]
            numList.append(int(num))
            #print(numList)
        except IndexError:
            pass
    numList = sorted(numList)
    newNum = numList[-1]+1
    saveName = 'gen_songs/generated_song%01d.wav' % newNum
    return saveName
