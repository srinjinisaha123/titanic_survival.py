import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)

    # Handle missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # Drop unnecessary columns
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

    # Features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    return X, y
