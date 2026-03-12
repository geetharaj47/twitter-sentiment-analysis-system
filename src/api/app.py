from fastapi import FastAPI
from pydantic import BaseModel
from src.transformer.predict_bert import predict

app = FastAPI()


class Tweet(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict_sentiment(tweet: Tweet):

    sentiment = predict(tweet.text)

    return {
        "tweet": tweet.text,
        "sentiment": sentiment
    }