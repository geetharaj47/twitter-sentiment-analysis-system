import re
import pandas as pd
def clean_tweet(text):

    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def preprocess_dataset(input_path, output_path):

    df = pd.read_csv(input_path)

    df["clean_text"] = df["text"].apply(clean_tweet)

    df = df[["clean_text", "target"]]

    df.to_csv(output_path, index=False)

    print("Preprocessing completed.")

if __name__ == "__main__":

    input_path = "data/processed/sample_tweets.csv"
    output_path = "data/processed/cleaned_tweets.csv"

    preprocess_dataset(input_path, output_path)