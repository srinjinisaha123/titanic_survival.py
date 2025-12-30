import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from uploaded CSV (download train.csv from Kaggle and upload to repo)
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    st.error("train.csv not found. Please upload it to the repository.")
    st.stop()

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Titanic Survival Prediction AI")
st.write("Enter passenger details to predict survival.")
st.write(f"Model Accuracy on Test Data: {accuracy:.2f}")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.slider("Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 10.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Convert inputs
sex_num = 0 if sex == "Male" else 1
embarked_num = {"S": 0, "C": 1, "Q": 2}[embarked]

# Predict
input_data = np.array([[pclass, sex_num, age, sibsp, parch, fare, embarked_num]])
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

st.write(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
st.write(f"Survival Probability: {prob:.2f}")

# Visualization
st.subheader("Feature Importance")
importances = model.feature_importances_
features = X.columns
fig, ax = plt.subplots()
ax.barh(features, importances)
st.pyplot(fig)
