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
def fun():
	st.header("placement prediction")
	st.info("enter all the details")

	age=st.number_input("age")
	gen=st.radio("enter gender",["male","female"])
	stream=st.radio("enter stream",["cse","ece","mech"])
	interns=st.number_input("how many internships:")
	cgpa=st.number_input("enter CGPA:")
	back=st.number_input("enter the no backlogs")
	if gen=="Male":
		gen=1
	else:
		gen=0

	if stream=="cse":
		stream=1
	elif stream=="ece":
		stream=4
	elif stream=="mech":
		stream=5
	else:
		stream=2




	li=[age,gen,stream,interns,cgpa,back]
	x=st.button("submit")
	if x:
		output=model.predict([li])
		if output==1:
			st.success("yes you got placement")
		else:
			st.error("no you are not placed")






fun()
