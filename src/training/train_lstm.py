import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

from src.models.lstm_model import SentimentLSTM


MAX_LEN = 50
BATCH_SIZE = 64
EMBED_DIM = 200
HIDDEN_DIM = 128
OUTPUT_DIM = 2
EPOCHS = 5


def build_vocab(texts, max_size=20000):

    counter = Counter()

    for text in texts:
        counter.update(text.split())

    most_common = counter.most_common(max_size)

    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}

    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    return vocab


def encode(text, vocab):

    tokens = text.split()

    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
    else:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

    return ids


class TweetDataset(Dataset):

    def __init__(self, texts, labels, vocab):

        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoded = encode(self.texts[idx], self.vocab)

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


if __name__ == "__main__":

    df = pd.read_csv("data/processed/cleaned_tweets.csv")

    df = df.dropna()

    # optional speed-up
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

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    vocab = build_vocab(X_train)

    train_dataset = TweetDataset(X_train, y_train, vocab)
    test_dataset = TweetDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentLSTM(
        vocab_size=len(vocab) + 2,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(true_labels, predictions)

    print("LSTM Accuracy:", accuracy)