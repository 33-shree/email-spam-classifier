
import streamlit as st
import pickle
import sklearn

# Load the saved model and vectorizer
model = pickle.load(open("spam_classifier_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", layout="centered")
st.title("📧 Email Spam Classifier")
st.write("Enter a message below and click **Predict** to know whether it's spam or not.")

# User input
message = st.text_area("Enter your email or message:")

# Predict button
proba = model.predict_proba(input_data)[0]
st.write(f"Prediction Probabilities: {proba}")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Transform and predict
        input_data = vectorizer.transform([message])
        result = model.predict(input_data)[0]

        if result == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")
