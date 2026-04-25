import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
 'resume':['python ml project','sales marketing','data science python','accounting finance'],
 'role':['tech','non-tech','tech','non-tech']
}

df = pd.DataFrame(data)

vec = TfidfVectorizer()
X = vec.fit_transform(df['resume'])
y = df['role']

model = LogisticRegression().fit(X,y)

print(model.predict(vec.transform(['machine learning python'])))