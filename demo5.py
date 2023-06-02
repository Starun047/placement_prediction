import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#Step 1
#Loading the dataset:

df= pd.read_csv("dataset1.csv")


#Step 2
#Data preproccessing

le=LabelEncoder()
stream=le.fit_transform(df['Stream'])
df["Stream"]=stream
x=df.pop("Stream")
df.insert(2,"Stream",x)


x=le.fit_transform(df["Gender"])
df.drop("Gender",axis=1,inplace=True)
df.insert(1,"Gender",x)

#Step 3
#Building our model

x_train,x_test,y_train,y_test=train_test_split(df[['Age','Gender','Stream','Internships','CGPA','HistoryOfBacklogs']],df.PlacedOrNot,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


#Step 4
#Building Streamlit app
