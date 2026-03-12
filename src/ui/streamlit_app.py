import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Twitter Sentiment Analysis")

st.write("Enter a tweet and the model will predict sentiment.")

tweet = st.text_area("Enter Tweet")

if st.button("Predict Sentiment"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet")
    else:
        response = requests.post(
            API_URL,
            json={"text": tweet}
        )

        result = response.json()

        st.success(f"Sentiment: {result['sentiment']}")