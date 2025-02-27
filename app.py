import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

#load dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.keras')

def pre_processing(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

st.title('IMDB user review Sentiment analysis.')
st.write('Enter the user review:')

user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = pre_processing(user_input)

    # Make prediction:
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.97 else 'Negative'
    st.write(sentiment)
else:
    st.write('Please enter the movie review.')
