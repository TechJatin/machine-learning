import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
 'study':[1,2,3,4],
 'score':[30,40,55,70]
})

model = LinearRegression().fit(df[['study']], df['score'])
print(model.predict([[5]]))