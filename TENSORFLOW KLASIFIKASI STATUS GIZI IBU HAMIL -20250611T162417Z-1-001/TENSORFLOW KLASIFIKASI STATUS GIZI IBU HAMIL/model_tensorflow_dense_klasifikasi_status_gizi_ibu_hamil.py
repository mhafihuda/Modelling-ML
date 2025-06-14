# -*- coding: utf-8 -*-
"""MODEL TENSORFLOW DENSE - KLASIFIKASI STATUS GIZI IBU HAMIL

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dvwwvIOMSixYKSjxpLX6j6oRzvhUihN8
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from tensorflow.keras.models import load_model
import joblib

df = pd.read_csv("/content/dummy_status_gizi_bumil_new.csv")
df.head()

df.info()

df.nunique()

df.describe()

"""## Data Preparation"""

df_drop = df.drop(['no', 'umur', 'nama'], axis=1)
df_drop.head()

df_drop[['sistolik', 'diastolik']] = df_drop['tekanan'].str.split('/', expand=True).astype(int)
df_drop.drop(columns=['tekanan'], inplace=True)
df_drop.head()

df_drop.dtypes

label_encoder = LabelEncoder()
df_drop['status_gizi_encoded'] = label_encoder.fit_transform(df_drop['status_gizi'])


# Fitur numerik
features = ['bb_dulu', 'bb_sekarang', 'tinggi_badan', 'lila', 'hb', 'IMT', 'sistolik', 'diastolik']
X = df_drop[features]
y = df_drop['status_gizi_encoded']

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split: train 70%, val 15%, test 15%
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

print("Jumlah data train:", len(X_train))
print("Jumlah data validation:", len(X_val))
print("Jumlah data test:", len(X_test))

"""## modeling TensorFlow"""

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi di data test: {test_accuracy:.2f}")

"""SAVED MODEL"""

# Simpan model dalam format HDF5
model.save('model_status_gizi.h5')

# Atau format baru .keras (SavedModel)
model.save('model_status_gizi.keras')