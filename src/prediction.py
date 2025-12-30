import numpy as np

def predict_survival(model, passenger_data):
    passenger = np.array([passenger_data])
    prediction = model.predict(passenger)

    return "Survived" if prediction[0] == 1 else "Did Not Survive"
