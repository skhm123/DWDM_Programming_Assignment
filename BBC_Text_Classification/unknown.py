import pandas as pd
import string
df=pd.read_csv(r"bbc-text.csv",encoding='latin-1')
l=df.iloc[:,0]
df=df.iloc[:,1]
df=df.str.lower()
import nltk
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
b=[w for w in df if not w in stop]
df=pd.DataFrame(b)
d=''
a=[]
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
for j in range(len(df)):
	for i in df.iloc[j,:]:
		for w in i:
			if w not in punctuations:
				d=d+w
		a.append(d)
		d=''
df=pd.DataFrame(a)
d=''
a=[]
for j in range(len(df)):
	for i in df.iloc[j,:]:
		for w in i:
			if not w.isdigit():
				d=d+w
		a.append(d)
		d=''
df=a
from nltk.stem import PorterStemmer
for i in range(len(df)):
	df[i]=PorterStemmer().stem(df[i])
df=pd.DataFrame(df)
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x = v.fit_transform(df.iloc[:,0])
df=x.toarray()
e=[]
for i in l:
	if i == 'business':
		e.append(0)
	elif i == 'entertainment':
		e.append(1)
	elif i == 'politics':
		e.append(2)
	elif i == 'sport':
		e.append(3)
	elif i == 'tech':
		e.append(4)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,e,test_size=0.2,random_state=46)
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
reg=DecisionTreeClassifier()
reg.fit(x_train,y_train)
y=reg.predict(x_test)
print(accuracy_score(y_test,y))

from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier()
reg.fit(x_train,y_train)
y=reg.predict(x_test)
print(accuracy_score(y_test,y))

from sklearn.naive_bayes import GaussianNB
reg=GaussianNB()
reg.fit(x_train,y_train)
y=reg.predict(x_test)
print(accuracy_score(y_test,y))

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
y=reg.predict(x_test)

print(accuracy_score(y_test,y))