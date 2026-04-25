import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'hour':[1,2,3,4,5],
    'traffic':[50,80,120,200,300]
})

model = LinearRegression().fit(df[['hour']], df['traffic'])

print("Traffic at hour 6:", model.predict([[6]]))