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

# path dos dados
train_data = pd.read_csv("./data/Metadata_Train.csv", header=0)
test_data = pd.read_csv("./data/Metadata_Test.csv", header=0)


def pad_features(features, max_length):
    if features.shape[1] < max_length:
        return np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant', constant_values=0)
    else:
        return features[:, :max_length]

# features
max_features_length = 0

def extract_features(file_path):
    global max_features_length  # limita o tamanho
    y, sr = librosa.load(file_path, duration=2.97)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # correção
    max_features_length = max(max_features_length, mfccs.shape[1])
    
    # pad das features
    mfccs_padded = pad_features(mfccs, max_features_length)
    
    return mfccs_padded

# features pro treino
train_data['Features'] = train_data['FileName'].apply(lambda x: extract_features("./data/Train_submission/" + x))

# features pro teste
test_data['Features'] = test_data['FileName'].apply(lambda x: extract_features("./data/Test_submission/" + x))

# classes
label_encoder = LabelEncoder()
train_data['Class'] = label_encoder.fit_transform(train_data['Class'])
test_data['Class'] = label_encoder.transform(test_data['Class'])

# pad nas características de treino
train_data['Features'] = train_data['Features'].map(lambda x: pad_features(x, max_features_length))

# pad nas características de teste
test_data['Features'] = test_data['Features'].map(lambda x: pad_features(x, max_features_length))

# treino -> validação
X_train, X_val, y_train, y_val = train_test_split(
    np.stack(train_data['Features']), train_data['Class'], test_size=0.2
)

# verifica o formato atual de X_train
print("Formato atual de X_train:", X_train.shape)

# reshape pto cnn
X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])

# modelo CNN
model = tf.keras.Sequential()
model.add(keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=(128, 13)))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Conv1D(128, kernel_size=6, activation='tanh'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.2))  #dropout
model.add(keras.layers.Dense(len(label_encoder.classes_), activation='softmax'))


# model.add(keras.layers.Dropout(0.2))


# compila modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


# reshape pro formato correto
X_train = X_train.transpose(0, 1, 2)
X_val = X_val.transpose(0, 1, 2)

# treina o modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# pre teste
X_test = np.stack(test_data['Features']).reshape(test_data.shape[0], max_features_length, 13)
y_test = test_data['Class'].to_numpy()

# previsões
y_pred = model.predict(X_test)

# previsões > labels
y_pred_classes = np.argmax(y_pred, axis=1)

# acuracia
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Acurácia nos dados de teste: {accuracy}")


# # avaliação
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Acurácia nos dados de teste: {test_accuracy}")



# mostrar o espectrograma
def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=2.97)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    plt.show()


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
# new_audio_path = "data/demo/trans-la.wav" #NOTA LA
# new_audio_path = "data/Test_submission/rock-drum-loop-85371.wav" #erro (violino)
# new_audio_path = "data/Test_submission/acoustic-guitar-logo-13084.wav" #erro (violino)
# new_audio_path = "data/Test_submission/guitar-chords-70663.wav" #erro (bateria)
# new_audio_path = "data/Test_submission/wingrandpiano-96338.wav" #erro (bateria)
# new_audio_path = "data/Test_submission/hip-hop-drum-loop-22-33572.wav" #ok
new_audio_path = "data/Test_submission/Sad-Violin-Slow-K-www.fesliyanstudios.com.wav" #ok









new_audio_features = extract_and_pad_features(new_audio_path, max_features_length)

new_audio_features = new_audio_features.transpose(1, 0)

new_audio_prediction = model.predict(np.expand_dims(new_audio_features, axis=0))

predicted_class = np.argmax(new_audio_prediction)

predicted_class_original = label_encoder.inverse_transform([predicted_class])[0]




print(f"O {new_audio_path} é previsto como: {predicted_class_original}")


plot_spectrogram(new_audio_path)
plot_audio_features(new_audio_path, feature_type='waveform', duration=2.97)