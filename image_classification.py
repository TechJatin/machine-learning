# Handwritten Digit Classification

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

digits = load_digits()

X_train,X_test,y_train,y_test = train_test_split(
    digits.data, digits.target, test_size=0.2
)

model = SVC()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))
print("Prediction:", model.predict([X_test[0]]))