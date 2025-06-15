## Import Libraries 
import streamlit as st 
import numpy as np
import tensorflow as tf
import pickle 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

## Loading our LSTM Model 
model = load_model("next_word_pred_model.h5")

## Loading our tokenizer 
with open("tokenizer.pickle","rb") as handle :
    tokenizer = pickle.load(handle)

## Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

## Streamlit web app
st.title(":rainbow[Next Word Prediction Model]")
st.subheader(":grey[using LSTM RNN]")

## user input
input_text = st.text_area("Enter the sequence of words","I am good")

## Prediction 
if st.button("Predict") :
    max_sent_len = model.input_shape[1] + 1 ## max sequence length
    next_word_pred = predict_next_word(model,tokenizer,input_text,max_sent_len) 
    st.success(f"Next predicted word = {next_word_pred}")
 