import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

# dados
train_data = pd.read_csv("./data/Metadata_Train.csv", header=0)
test_data = pd.read_csv("./data/Metadata_Test.csv", header=0)

# Defina primeiro a função pad_features
def pad_features(features, max_length):
    if features.shape[1] < max_length:
        return np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant', constant_values=0)
    else:
        return features[:, :max_length]

# features
max_features_length = 0  # Inicializa max_features_length

def extract_features(file_path):
    global max_features_length  # Usa a variável global
    y, sr = librosa.load(file_path, duration=2.97)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Atualiza max_features_length se necessário
    max_features_length = max(max_features_length, mfccs.shape[1])
    
    # Ajusta o formato para garantir que todos os arrays tenham a mesma forma
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

# Pad nas características de treino
train_data['Features'] = train_data['Features'].map(lambda x: pad_features(x, max_features_length))

# Pad nas características de teste
test_data['Features'] = test_data['Features'].map(lambda x: pad_features(x, max_features_length))

# treino -> validação
X_train, X_val, y_train, y_val = train_test_split(
    np.stack(train_data['Features']), train_data['Class'], test_size=0.2, random_state=42
)

# Adiciona esta linha de impressão para verificar o formato atual de X_train
print("Formato atual de X_train:", X_train.shape)

# reshape pto cnn
X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])

# modelo CNN
model = tf.keras.Sequential()
model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(128, 13)))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(len(label_encoder.classes_), activation='softmax'))

# compila modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


# Redimensiona para o formato correto
X_train = X_train.transpose(0, 1, 2)
X_val = X_val.transpose(0, 1, 2)

# Treina o modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# pre teste
X_test = np.stack(test_data['Features']).reshape(test_data.shape[0], max_features_length, 13)
y_test = test_data['Class'].to_numpy()

# Obter previsões no conjunto de teste
y_pred = model.predict(X_test)

# Converter previsões para rótulos de classe
y_pred_classes = np.argmax(y_pred, axis=1)

# acuracia
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Acurácia nos dados de teste: {accuracy}")


# # avaliação
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Acurácia nos dados de teste: {test_accuracy}")


