# Text Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = ["machine learning is fun", "AI is interesting", "machine learning is powerful"]

vec = TfidfVectorizer()
X = vec.fit_transform(texts)

similarity = cosine_similarity(X)

print(similarity)