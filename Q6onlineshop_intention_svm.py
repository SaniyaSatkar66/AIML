#Q6 
import joblib
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df= pd.read_csv('online_shoppers_intention.csv')



label_encoder={}
for col in['Month','VisitorType','Weekend']:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    label_encoder[col]=le



df.dropna(inplace=True)

X=df.drop('Revenue',axis=1)
y=df['Revenue']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.39,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svm_model=SVC(kernel='linear')
svm_model.fit(X_train,y_train)

joblib.dump(svm_model,'svm_model7.pkl')
joblib.dump(scaler,'scaler7.pkl')
print("model saved")

loaded_model=joblib.load('svm_model7.pkl')
loaded_scaler=joblib.load('scaler7.pkl')

y_pred=svm_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy score of model:{accuracy *100:.2f}")