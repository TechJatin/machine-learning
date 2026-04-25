# Sales Prediction

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'advertising':[10,20,30,40,50],
    'sales':[100,150,200,250,300]
})

X = df[['advertising']]
y = df['sales']

model = LinearRegression()
model.fit(X,y)

print("Predicted Sales:", model.predict([[35]]))