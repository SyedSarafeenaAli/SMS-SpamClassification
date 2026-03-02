import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK data exists
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered_tokens = []
    for word in tokens:
        if word.isalnum():
            filtered_tokens.append(word)

    cleaned_tokens = []
    stop_words = set(stopwords.words("english"))

    for word in filtered_tokens:
        if word not in stop_words and word not in string.punctuation:
            cleaned_tokens.append(word)

    stemmed_tokens = [ps.stem(word) for word in cleaned_tokens]

    return " ".join(stemmed_tokens)

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "text"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["transformed"] = df["text"].apply(transform_text)

# Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["transformed"])
y_labels = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y_labels)

# Save trained objects
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")