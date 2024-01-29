import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers, models

from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

# dados
train_data = pd.read_csv("./data/Metadata_Train.csv")
test_data = pd.read_csv("./data/Metadata_Test.csv")

# features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=2.97)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

max_features_length = max(train_data['Features'].apply(lambda x: x.shape[1]).max(),
                          test_data['Features'].apply(lambda x: x.shape[1]).max())

def pad_features(features, max_length):
    return np.pad(features, ((0, 0), (0, max_length - features.shape[1])))


# features pro treino
train_data['Features'] = train_data['FileName'].apply(lambda x: extract_features("./data/Train_submission/" + x))

# features pro teste
test_data['Features'] = test_data['FileName'].apply(lambda x: extract_features("./data/Test_submission/" + x))

# classes
label_encoder = LabelEncoder()
train_data['Class'] = label_encoder.fit_transform(train_data['Class'])
test_data['Class'] = label_encoder.transform(test_data['Class'])

# treino -> validação
X_train, X_val, y_train, y_val = train_test_split(
    np.vstack(train_data['Features']), train_data['Class'], test_size=0.2, random_state=42
)

# reshape pto cnn
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# modelo CNN
model = models.Sequential()
model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# compila modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# treina o modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)


# pre teste
X_test = test_data['Features'].to_numpy().reshape(test_data.shape[0], X_train.shape[1], 1)
y_test = test_data['Class'].to_numpy()

# avaliaçãp
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia nos dados de teste: {test_accuracy}")
