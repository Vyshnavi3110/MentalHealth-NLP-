# app.py
import streamlit as st
import os
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# ------------------ NLTK Setup ------------------
nltk_data_dir = "/tmp/nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download required NLTK packages if missing
for pkg in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)

# Load stopwords and lemmatizer
stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# ------------------ Check required files ------------------
required_files = ["mental_health_model.h5", "tfidf_vectorizer.pkl", "label_encoder.pkl"]
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Missing file: {file}. Please upload it to the repo.")
        st.stop()

# ------------------ Load model and pickles ------------------
model = load_model("mental_health_model.h5")

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ------------------ Text preprocessing ------------------
def all_inone(text):
    """Clean, tokenize, remove stopwords, and lemmatize the text."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)     # remove non-letters
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop]
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
    return " ".join(tokens)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Text Classifier")
st.write("Enter any text below, and the model will predict the mental health class.")

# User input
user_input = st.text_area("Type your text here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = all_inone(user_input)
        vec = tfidf.transform([cleaned]).toarray()
        pred = np.argmax(model.predict(vec), axis=1)
        label = le.inverse_transform(pred)[0]
        st.success(f"**Predicted Class:** {label}")
    else:
        st.warning("Please enter some text to predict!")

# Optional: Reset button for multiple predictions
if st.button("Clear"):
    st.experimental_rerun()
