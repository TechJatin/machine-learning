# Loan Approval Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'income':[25000,40000,50000,30000,60000,70000,20000],
    'loan_amount':[10000,20000,25000,12000,30000,35000,8000],
    'credit_score':[600,700,750,650,800,820,580],
    'approved':[0,1,1,0,1,1,0]
}

df = pd.DataFrame(data)

# Features
X = df[['income','loan_amount','credit_score']]
y = df['approved']

# Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Model
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))

# Prediction function
def predict_loan(income,loan,credit):
    res = model.predict([[income,loan,credit]])
    return "Approved" if res[0]==1 else "Rejected"

print(predict_loan(45000,15000,720))