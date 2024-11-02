import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import streamlit as st

nltk.download('punkt_tab')
nltk.download('wordnet')

st.set_page_config(page_title="chatbot", layout="wide")

st.title("🤖 Asistente virtual")

# Cargar el modelo y datos
lemmatizer = WordNetLemmatizer()

# Cargar datos del archivo JSON y el modelo entrenado
with open('D:/users/kevin/phyton/chat_bot/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

model = load_model('d:/users/kevin/phyton/chat_bot/chatbot_model.h5')

# Preparar las palabras y clases
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Función para preprocesar la entrada del usuario
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Función para convertir la entrada del usuario en un vector de características
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

# Función para predecir la clase de la entrada del usuario
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Función para obtener la respuesta correspondiente a la clase predicha
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Función para interactuar con el usuario
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

# Interfaz Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mensaje de bienvenida del asistente
if st.session_state.first_message:
    welcome_message = "hola, ¿como puedo ayudarte?"
    with st.chat_message("assistant"):
        st.markdown(welcome_message)
    
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    st.session_state.first_message = False

# Procesar la entrada del usuario
if prompt := st.chat_input("¿como puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chatbot_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
