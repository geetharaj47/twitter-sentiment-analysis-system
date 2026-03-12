import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments


# Load dataset
df = pd.read_csv("data/processed/cleaned_tweets.csv")
df = df.dropna()

texts = df["clean_text"].tolist()
labels = df["target"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(texts):

    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128
    )


train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)


class TweetDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):

        return len(self.labels)


train_dataset = TweetDataset(train_encodings, y_train)
test_dataset = TweetDataset(test_encodings, y_test)


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)


def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions)
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()


print("BERT Accuracy:", results["eval_accuracy"])