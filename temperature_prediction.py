# Temperature Prediction (Time-based)

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'day':[1,2,3,4,5],
    'temp':[30,32,34,33,35]
})

X = df[['day']]
y = df['temp']

model = LinearRegression()
model.fit(X,y)

print("Predicted Temp:", model.predict([[6]]))