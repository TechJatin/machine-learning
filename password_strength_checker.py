import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    'password':['12345','abc123','Strong@123','pass','Admin@2024'],
    'strength':[0,0,2,0,2]  # 0=weak,2=strong
}

df = pd.DataFrame(data)

vec = TfidfVectorizer(analyzer='char')
X = vec.fit_transform(df['password'])
y = df['strength']

model = LogisticRegression().fit(X,y)

print(model.predict(vec.transform(['MyPass@123'])))