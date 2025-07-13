import streamlit as st
import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Load and preprocess data
df = pd.read_csv("sentiment.csv")
df['text'] = df['text'].apply(clean_text)

# Tokenization
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)
y = df['label'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build RNN model
model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_len))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
with st.spinner("Training model..."):
    model.fit(X_train, y_train, epochs=5, batch_size=2, validation_split=0.1, verbose=0)

# Streamlit UI
st.title("🧠 Sentiment Analysis - RNN Model")
st.write("Enter a movie review below and the model will predict if it's **Positive** or **Negative**.")

user_input = st.text_area("🎬 Enter your review here", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to predict.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len)
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative 😞"
        st.markdown(f"### Result: {sentiment}")
        st.markdown(f"**Confidence:** {prediction:.2f}")
