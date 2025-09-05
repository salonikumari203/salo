import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

print("Downloading dataset (IMDB reviews)...")
dataset = fetch_openml("IMDB", version=1, as_frame=True)

df = dataset.frame
df = df.rename(columns={"review": "text", "sentiment": "label"})

print("Dataset sample:\n", df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=200, solver="liblinear")
model.fit(X_train_vec, y_train
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

examples = [
    "This movie was fantastic! Great acting and wonderful story.",
    "I hated this movie. The plot was boring and the acting was terrible.",
    "It was okay, not the best but not the worst either."
]

example_vec = vectorizer.transform(examples)
preds = model.predict(example_vec)

print("\nSample Predictions:")
for text, p in zip(examples, preds):
    print(f"Text: {text}\n → Sentiment: {p}
sent_counts = pd.Series(y_pred).value_counts()
sent_counts.plot(kind="bar", title="Sentiment Distribution in Test Predictions")
plt.xticks(rotation=0)
plt.show()






