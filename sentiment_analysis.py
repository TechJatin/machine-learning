# Sentiment Analysis (Better Version)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'text':[
        'I love this product',
        'Worst experience ever',
        'Amazing quality and service',
        'Very bad and disappointing',
        'I am happy with this',
        'Not good at all'
    ],
    'sentiment':[1,0,1,0,1,0]
}

df = pd.DataFrame(data)

X_train,X_test,y_train,y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.3, random_state=42
)

vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec,y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test_vec)))

# Custom input
def predict_sentiment(text):
    return "Positive" if model.predict(vec.transform([text]))[0]==1 else "Negative"

print(predict_sentiment("I love it so much"))