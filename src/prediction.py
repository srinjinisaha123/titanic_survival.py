import numpy as np

def predict_survival(model, passenger_data):
    passenger_array = np.array([passenger_data])
    prediction = model.predict(passenger_array)

    if prediction[0] == 1:
        return "Survived"
    else:
        return "Did Not Survive"
