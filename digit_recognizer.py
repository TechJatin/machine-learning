from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

data = load_digits()
model = KNeighborsClassifier().fit(data.data, data.target)

print(model.predict([data.data[0]]))