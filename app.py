import streamlit as st
import pickle
import numpy as np
import re
import warnings
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# -------------------- SUPPRESS WARNINGS --------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------- NLTK SETUP --------------------
# Create a local nltk_data folder for Streamlit Cloud
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("omw-1.4", download_dir=nltk_data_dir)  # WordNet support for multiple languages

stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# -------------------- LOAD MODELS --------------------
model = load_model("mental_health_model.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------- TEXT PREPROCESSING --------------------
def preprocess_text(text):
    """Clean and tokenize any user input safely."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)      # Remove non-alphabetic characters
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()  # fallback if punkt is missing
    tokens = [t for t in tokens if t not in stop]
    try:
        tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    except LookupError:
        pass  # fallback: skip lemmatization if WordNet is missing
    return " ".join(tokens)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠")
st.title("🧠 Mental Health Text Classifier")
st.write("Enter any text and get the predicted mental health class.")

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
