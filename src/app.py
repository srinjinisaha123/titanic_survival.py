from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.prediction import predict_survival

# Load and preprocess data
X, y = load_and_preprocess_data("data/titanic.csv")

# Train model
model, accuracy, report = train_model(X, y)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Test prediction
# [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
sample_passenger = [1, 1, 25, 0, 0, 100, 2]

result = predict_survival(model, sample_passenger)
print("\nPrediction for sample passenger:", result)
