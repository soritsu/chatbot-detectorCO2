import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Cargar datos del archivo JSON
with open('D:/users/kevin/phyton/chat_bot/intents.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Procesar cada intención
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenizar cada palabra
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatizar y convertir a minúsculas, eliminar palabras ignoradas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

classes = sorted(set(classes))

# Crear nuestros datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

# Verificar la consistencia de las sublistas
for i, item in enumerate(training):
    if len(item[0]) != len(words) or len(item[1]) != len(classes):
        print(f"Error en el ítem {i}: {item}")
        raise ValueError("Las sublistas no tienen la longitud correcta")

# Convertir a np.array
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]), dtype=float)
train_y = np.array(list(training[:, 1]), dtype=float)

print("Datos de entrenamiento procesados correctamente.")

# Crear el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot_model.h5')

print("Modelo creado y guardado correctamente.")
