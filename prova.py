
from google.colab import drive
drive.mount('/content/drive')

import os

print("Conteúdo de MyDrive:", os.listdir('/content/drive/MyDrive'))

prova_path = '/content/drive/MyDrive/Prova'
print("Conteúdo da pasta Prova:", os.listdir(prova_path))

imagens_path = prova_path + '/imagens'
print("Conteúdo da pasta imagens:", os.listdir(imagens_path))

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# --- CONFIG ---
IMG_SIZE_CIFAR = (32, 32)
IMG_SIZE_LOCAL = (128, 128)
BATCH_SIZE = 64
EPOCHS = 15

# Caminho da pasta imagens no Google Drive (gatos e cachorros)
DATA_DIR = '/content/drive/MyDrive/Prova/imagens'

# Classes usadas para imagens locais
local_class_names = ['gato', 'cachorro']

# 1) Carrega CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normaliza para [0,1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Modelo CNN para CIFAR-10 (input 32x32)
model_cifar = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE_CIFAR + (3,)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes CIFAR-10
])

model_cifar.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

print("Treinando modelo CIFAR-10...")
history_cifar = model_cifar.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

test_loss, test_acc = model_cifar.evaluate(x_test, y_test, verbose=2)
print(f"\nAcurácia no teste CIFAR-10: {test_acc:.4f}")

# --- PREPROCESSAMENTO IMAGENS LOCAIS ---

def preprocess_image(path):
    # Carrega imagem colorida
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Imagem não carregada: {path}")
    # Redimensiona para 128x128
    img = cv2.resize(img, IMG_SIZE_LOCAL)
    # Aplica filtro gaussiano
    img = cv2.GaussianBlur(img, (5,5), 0)
    # Converte para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equaliza histograma
    equalized = cv2.equalizeHist(gray)
    # Para usar no modelo CNN precisamos de 3 canais, então replicamos o canal
    processed = cv2.merge([equalized, equalized, equalized])
    # Normaliza [0,1]
    processed = processed.astype('float32') / 255.0
    return processed

# 2) Carrega imagens locais (gatos e cachorros)
images = []
labels = []

for idx, cls in enumerate(local_class_names):
    folder = os.path.join(DATA_DIR, cls)
    if not os.path.exists(folder):
        print(f"[AVISO] Pasta não encontrada: {folder}")
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, fname)
            try:
                img = preprocess_image(path)
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(e)

images = np.array(images)
labels = np.array(labels)

print(f"Imagens carregadas: {images.shape[0]}")

# 3) Separa treino/teste 80% / 20%
x_train_local, x_test_local, y_train_local, y_test_local = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)

# 4) Define modelo CNN para as imagens locais (128x128)
model_local = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE_LOCAL + (3,)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(local_class_names), activation='softmax')
])

model_local.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

print("\nTreinando modelo local (gatos e cachorros)...")
history_local = model_local.fit(
    x_train_local, y_train_local,
    epochs=EPOCHS,
    batch_size=8,
    validation_split=0.2
)

# 5) Avalia modelo local
y_pred_local = np.argmax(model_local.predict(x_test_local), axis=1)

print("\nRelatório de classificação (gatos e cachorros):")
print(classification_report(y_test_local, y_pred_local, target_names=local_class_names))

# 6) Função para predizer com modelo CIFAR-10 nas imagens locais (32x32)
def prediz_local_cifar(modelo, pasta, img_size=(32,32)):
    print("\nPredições do modelo CIFAR-10 em imagens locais:")
    for cls in local_class_names:
        folder = os.path.join(pasta, cls)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(folder, fname)
            img = image.load_img(path, target_size=img_size)
            arr = image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, 0)
            probs = modelo.predict(arr, verbose=0)[0]
            pred_idx = np.argmax(probs)
            conf = probs[pred_idx]
            print(f"{cls}/{fname:20} → {class_names[pred_idx]} (Confiança={conf:.2f})")

# Prediz imagens locais com modelo CIFAR-10
prediz_local_cifar(model_cifar, DATA_DIR)

