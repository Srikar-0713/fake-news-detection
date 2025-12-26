import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("news.csv")

texts = data["text"]
labels = data["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

test_news = [
    "New education policy announced by government",
    "Drinking salt water can cure cancer"
]

test_vectors = vectorizer.transform(test_news)
predictions = model.predict(test_vectors)

for news, result in zip(test_news, predictions):
    print(news)
    print("Prediction:", result)
    print()
