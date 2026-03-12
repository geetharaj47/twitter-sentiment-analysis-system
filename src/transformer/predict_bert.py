import torch
from transformers import AutoTokenizer, BertForSequenceClassification

MODEL_PATH = "src/models/bert_sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    return "Positive 😀" if prediction == 1 else "Negative 😞"


if __name__ == "__main__":

    text = input("Enter a tweet: ")

    result = predict(text)

    print("Sentiment:", result)