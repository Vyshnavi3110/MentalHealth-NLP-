import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords and lemmatizer
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Load saved model, TF-IDF vectorizer, and LabelEncoder
model = load_model("mental_health_model.h5")

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Preprocessing function
def all_inone(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("🧠 Mental Health Text Classifier")
st.write("Enter a sentence and get the predicted mental health class.")

# Text input
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = all_inone(user_input)
        vec = tfidf.transform([cleaned]).toarray()
        pred = np.argmax(model.predict(vec), axis=1)
        label = le.inverse_transform(pred)[0]

        st.success(f"Predicted Class: **{label}**")
    else:
        st.warning("Please enter some text to predict!")
