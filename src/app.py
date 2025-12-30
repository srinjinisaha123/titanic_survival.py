import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Unsinkable AI Pilot",
    page_icon="🚢",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🚢 Unsinkable AI Pilot")
st.subheader("Titanic Survival Prediction using AI")

st.write(
    "This AI model predicts whether a Titanic passenger survived "
    "based on historical data and machine learning."
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

data = load_data()

# ---------------- PREPROCESS ----------------
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])

X = data.drop('Survived', axis=1)
y = data['Survived']

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model()

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

st.markdown("---")

# ---------------- USER INPUT ----------------
st.header("Passenger Information")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
fare = st.number_input("Ticket Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encoding inputs
sex = 1 if sex == "Female" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

# ---------------- PREDICTION ----------------
if st.button("Predict Survival"):
    passenger = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(passenger)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
