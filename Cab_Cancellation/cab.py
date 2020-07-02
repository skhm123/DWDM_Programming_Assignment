import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv(r"Kaggle_YourCabs_training.csv")
ans=df.iloc[:,18]
df=df.iloc[:,0:18]
df=df.fillna(df.mean())
df=df.drop(["from_date","to_date","booking_created"],axis=1)
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler()
df=scaler.fit_transform(df)
ans=np.asarray(ans)
x_train,x_test,y_train,y_test=train_test_split(df,ans,test_size=0.34,random_state=34)
from sklearn.naive_bayes import MultinomialNB
reg=MultinomialNB()
reg.fit(x_train,y_train)
d=reg.predict(x_test)
print(y_test)
print(d)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,d))
from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier()
reg.fit(x_train,y_train)
d=reg.predict(x_test)
print(accuracy_score(y_test,d))
