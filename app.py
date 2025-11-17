import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival", layout="centered")
st.title("Titanic Survival Predictor")

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

X = df[["Pclass","Sex","Age","SibSp","Parch","Fare"]].copy()
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Age"] = X["Age"].fillna(X["Age"].median())
y = df["Survived"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

col1, col2 = st.columns(2)
pclass = col1.selectbox("Class", [1, 2, 3], index=0)
age = col2.slider("Age", 1, 80, 30)
fare = col1.slider("Fare", 0, 500, 50)
sex = col2.radio("Sex", ["Male", "Female"])

pred = model.predict([[pclass, 0 if sex=="Male" else 1, age, 0, 0, fare]])[0]
prob = model.predict_proba([[pclass, 0 if sex=="Male" else 1, age, 0, 0, fare]])[0][pred]

st.success("SURVIVED ✅" if pred else "Did not survive ❌")
st.progress(float(prob))
if pred:
    st.balloons()
