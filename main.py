
import os
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import sklearn

#sound api
import freesound, sys,os

#import keras

from tqdm import tqdm
import glob, os

from pathlib import Path





#loading data
def load_data():
    audio_path= 'train/positive/scream/1_scream_female_room.wav'
    data, sampling_rate = librosa.load(audio_path)

    #print(sampling_rate)
    #print(data)

    return data, sampling_rate



# waveplot
def save_waveplot(data,sampling_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr=sampling_rate)
    title= "Scream waveplot"
    plt.title(title)
    plt.savefig('plots/drafts/'+title, bbox_inches='tight')

    # Zooming in
    n0 = 9000
    n1 = 9100
    plt.figure(figsize=(14, 5))
    plt.plot(data[n0:n1])
    plt.grid()
    plt.savefig('plots/drafts/' + title+'_zero_crossing', bbox_inches='tight')

    #verifying amount of zero crosses
    zero_crossings = librosa.zero_crossings(data[n0:n1], pad=False)
    print('zero crossings: '+ str(sum(zero_crossings)))

    print('spectral centroid samples amount: ')
    spectral_centroids = librosa.feature.spectral_centroid(data, sr=sampling_rate)[0]
    print(spectral_centroids.shape)

    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    time = librosa.frames_to_time(frames)

    #Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(data, sr=sampling_rate, alpha=0.4)
    plt.plot(time, normalize(spectral_centroids), color='r')
    plt.savefig('plots/drafts/' + title + '_spectral_centroids', bbox_inches='tight')

    #spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(data + 0.01, sr=sampling_rate)[0]
    librosa.display.waveplot(data, sr=sampling_rate, alpha=0.4)
    plt.plot(time, normalize(spectral_rolloff), color='g')
    plt.savefig('plots/drafts/' + title + '_spectral_rolloff', bbox_inches='tight')


# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#mfcc feature (normalized)
def save_specshow(data,sampling_rate):
    n_mfcc = 40 #12 #40
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)


    #print(mfccs.shape)
    #print(type(mfccs))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    title= "Scream specshow mfccs=40"
    plt.title(title)
    plt.colorbar()
    plt.savefig('plots/drafts/'+title, bbox_inches='tight')

    #to the same with scaling
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    print(mfccs.mean(axis=1))
    print(mfccs.var(axis=1))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    title= "Scream specshow mfccs=40 with scale"
    plt.title(title)
    plt.colorbar()
    plt.savefig('plots/drafts/'+title, bbox_inches='tight')


def save_chroma_stft(data,sampling_rate):
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(data, sr=sampling_rate, hop_length=hop_length)
    plt.figure(figsize=(15, 5))
    title= "Scream chroma_stft normalized"
    plt.title(title)
    librosa.util.normalize(chromagram)
    #librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length)
    plt.savefig('plots/drafts/'+title, bbox_inches='tight')
    print(chromagram[6])

def save_rms(data,sampling_rate):
    n_mfcc = 40 #12 #40
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
    S, phase = librosa.magphase(mfccs)
    rms = librosa.feature.rms(S=S)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    title= "Scream rms "
    plt.title(title)
    plt.semilogy(rms.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis = 'log', x_axis = 'time')

    plt.savefig('plots/drafts/'+title, bbox_inches='tight')

def play_showing_data(data, sampling_rate):
    n_mfcc = 40
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
    print("mfccs printed:")
    print(type(mfccs))
    print(mfccs)
    print("size= "  + str(mfccs.size))
    print("shape: "  + str(mfccs.shape))

    print("mfcc[0] printed:")
    print(type(mfccs[0]))
    print(mfccs[0])
    print("size= "  + str(mfccs[0].size))
    print("shape: "  + str(mfccs[0].shape))


    index=0
    mfcc_means= []
    for frequncyRate in mfccs:
        mfcc_means.append(np.mean(frequncyRate))
        index+=1
        print("loop frequncyRate "+ str(index)+" , shape:" + str(frequncyRate.shape) + " e mean= "+ str(np.mean(frequncyRate)))

        #print("e= " + str(e) + "e mean= "+ str(np.mean(e)) )
    print("means list is of size " + str(len(mfcc_means)))
    print(mfcc_means)
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = scaler.fit((np.array(mfcc_means)).reshape(-1, 1))
    print("scaling tool details:")
    print(X)
    print("normalized data:")
    print(scaler.transform((np.array(mfcc_means)).reshape(-1, 1)))

def soundApi():
    print("soundApi")

    client = freesound.FreesoundClient()
    client.set_token("1TUvgpT0yJzOjjez0U5LrSLyWkwhWyJduZ3JK2YC", "token")

    results = client.text_search(query="scream",filter="type:wav duration:[0 TO 5]" ,fields="id,name")

    print(results)
    for result in results:
        #print(type(result)) # printes Sound class
        print(result)

        """
        lev:
        the method below is correct, but its not as simple- it appears i need to use
        some oath algorithm...for now i'm putting it aside. documented in word the situation,
          #result.retrieve(".",result.name+".wav")       
        """

def splitMyWaves(path, durationInSec):
    """pip install pydub"""

    durationInSec= int(durationInSec)
    os.chdir(path)
    for fileName in tqdm(glob.glob("*.wav")):
        #print(fileName)
        src = path + fileName
        #print(src)
        edges = librosa.effects.split(audio, top_db=40, frame_length=128, hop_length=32)

def flow():

    #creating csv

    #create header
    header = 'filename waveplot zero_crossings spectral_centroid spectral_rolloff chroma_stft rms'  #TODO lev-future_improvement edit/add to get better results
    fcc_amount= 40  # lev's initial value here was 40- this is the feature resolution- usually between 12-40
    for i in range(1, fcc_amount+1):
        header += f' mfcc_{i}'
    header += ' label'
    header = header.split() #  split by spaces as default
    print(header)

    #load features from each wav file- put inside the lines below as a function

    #reaching each wav file
    path_train= Path("train")
    for path_label in path_train.iterdir():
        #  print(path_label)  #  train\negative
        label= path_label.name
        print(label)
        for path_class in path_label.iterdir():
            true_class= path_class.name
            print(true_class)
            #  print(path_class)  #  train\negative\scream
            wave_file_paths= path_class.glob('**/*.wav')  #  <class 'generator'>
            #  print(type(wave_file_paths))
            for wav_path in wave_file_paths:
                #  print(type(wav_path))  #  <class 'pathlib.WindowsPath'>
                #  print(wav_path)  #  train\positive\scream\110142__ryding__scary-scream-4.wav
                #  print(wav_path.name)  #  110142__ryding__scary-scream-4.wav

                break  #TODO lev: will be removed when i'll finish to write the function


    #[x for x in p.iterdir() if x.is_dir()]
    #print(labels)
    #for label in labels:
        #our_classes= os.listdir(os.join(""))
        #for our_class in
        #path= "train\\"+label


#load *.wav positive *.wav negative into one instance with labels for each
#extract features
#normalize/scale
#pass to keras



if __name__ == "__main__":
#  main:
    flow()


    #splitMyWaves("C:\\Users\\tunik\\PycharmProjects\\AI_distressRecognition\\train\\negative\\scream\\", 5)




#data, sampling_rate= load_data()
#  save_waveplot(data,sampling_rate)
#  save_specshow(data,sampling_rate)
#save_chroma_stft(data,sampling_rate)
#save_rms(data,sampling_rate)
#  play_showing_data(data, sampling_rate)
# soundApi()


