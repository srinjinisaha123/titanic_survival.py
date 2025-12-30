import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (using a sample; replace with real Kaggle data for better results)
# For demo, we'll use a small synthetic dataset. In real use, load from CSV.
data = {
    'Pclass': [3, 1, 3, 1, 3, 3, 2, 3, 3, 2],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male'],
    'Age': [22, 38, 26, 35, 35, np.nan, 27, 19, 28, 31],
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 1, 1],
    'Parch': [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
    'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 11.13, 21.08, 7.85, 18.0],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'C', 'S'],
    'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

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
print(f"Model Accuracy: {accuracy:.2f}")

# Streamlit App
st.title("Titanic Survival Prediction AI")
st.write("Enter passenger details to predict survival.")

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

# Visualization (optional)
st.subheader("Feature Importance")
importances = model.feature_importances_
features = X.columns
plt.barh(features, importances)
st.pyplot(plt)
