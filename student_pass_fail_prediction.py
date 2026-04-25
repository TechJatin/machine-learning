import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    'hours':[1,2,3,4,5],
    'pass':[0,0,1,1,1]
})

X = df[['hours']]
y = df['pass']

model = LogisticRegression().fit(X,y)

print(model.predict([[2.5]]))