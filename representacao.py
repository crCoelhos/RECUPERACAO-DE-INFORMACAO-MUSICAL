import numpy as np
import pandas as pd
import tensorflow as tf

import librosa

import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

def pad_features(features, max_length):
    if features.shape[1] < max_length:
        return np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant', constant_values=0)
    else:
        return features[:, :max_length]


# mostrar o espectrograma
def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=2.97)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    plt.show()


def extract_features(file_path):
    global max_features_length  # limita o tamanho
    y, sr = librosa.load(file_path, duration=2.97)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # correção
    max_features_length = max(max_features_length, mfccs.shape[1])
    
    # pad das features
    mfccs_padded = pad_features(mfccs, max_features_length)
    
    return mfccs_padded

# mostrar time domain waveform
def plot_audio_features(file_path, feature_type='waveform', duration=None):

    y, sr = librosa.load(file_path, duration=duration)

    if feature_type == 'waveform':
        plt.figure(figsize=(14, 5))
        plt.plot(librosa.times_like(y), y)
        plt.title(f'Forma de Onda do arquivo: {file_path}')
        plt.show()

    else:
        return


def extract_and_pad_features(file_path, max_features_length):
    y, sr = librosa.load(file_path, duration=2.97)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_padded = pad_features(mfccs, max_features_length)
    return mfccs_padded

# new_audio_path = "data/demo/desejodemenina1-vida-vazia.wav" #MÚSICA BOA
# new_audio_path = "data/demo/harm-la4.wav" #NOTA LA GAITA
# new_audio_path = "data/demo/trans-la.wav" #NOTA LA FLAUTA TRANSVERSAL
# new_audio_path = "data/demo/band-la.wav" #NOTA LA BANDOLIN
# new_audio_path = "data/Test_submission/rock-drum-loop-85371.wav" #erro (violino)
# new_audio_path = "data/Test_submission/acoustic-guitar-logo-13084.wav" #erro (violino)
# new_audio_path = "data/Test_submission/guitar-chords-70663.wav" #erro (bateria)
# new_audio_path = "data/Test_submission/wingrandpiano-96338.wav" #erro (bateria)
new_audio_path = "data/Test_submission/hip-hop-drum-loop-22-33572.wav" #ok
# new_audio_path = "data/Test_submission/Sad-Violin-Slow-K-www.fesliyanstudios.com.wav" #ok




plot_spectrogram(new_audio_path)
plot_audio_features(new_audio_path, feature_type='waveform', duration=120)