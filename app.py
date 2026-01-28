import streamlit as st
import pickle
import numpy as np
import re
import warnings
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# -------------------- SUPPRESS WARNINGS --------------------
warnings.filterwarnings("ignore")  # Suppress sklearn warnings
import logging, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# -------------------- NLTK SETUP --------------------
nltk.download('punkt')
nltk.download('stopwords')
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# -------------------- LOAD MODELS --------------------
model = load_model("mental_health_model.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])  # Suppress absl warning

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------- TEXT PREPROCESSING --------------------
def preprocess_text(text):
    """Clean and tokenize any user input."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)      # Remove non-alphabetic characters
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    return " ".join(tokens)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠")
st.title("🧠 Mental Health Text Classifier")
st.write("Enter any sentence or text and get the predicted mental health class.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        cleaned = preprocess_text(user_input)
        vec = tfidf.transform([cleaned]).toarray()
        pred = np.argmax(model.predict(vec), axis=1)
        label = le.inverse_transform(pred)[0]
        st.success(f"Predicted Class: **{label}**")
    else:
        st.warning("Please enter some text to predict!")
