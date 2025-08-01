import streamlit as st
import joblib
import os
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Paths to model and vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Load model and vectorizer with joblib
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    tfidf = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model or vectorizer file not found. Please retrain and save them.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")
st.title("📧 Email Spam Classifier")
st.write("Detect whether a message is **Spam** or **Not Spam** using Machine Learning.")

# Input box
input_sms = st.text_area("Enter the message", height=150)

# Prediction
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)

        # 4. Display result
        if result == 1:
            st.error("🚨 Spam Detected!")
        else:
            st.success("✅ Not Spam")

        # 5. Show confidence scores
        st.caption(f"Confidence: Spam {proba[0][1]*100:.2f}%, Not Spam {proba[0][0]*100:.2f}%")
