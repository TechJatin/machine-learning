import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'experience':[1,2,3,4,5],
    'salary':[15000,20000,30000,40000,50000]
})

X = df[['experience']]
y = df['salary']

model = LinearRegression().fit(X,y)

print(model.predict([[3.5]]))