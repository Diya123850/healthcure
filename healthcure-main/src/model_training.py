import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)

# Load your data
data = pd.read_csv('data/sample_patient_data.csv')

# Preprocess, e.g., one-hot encoding for categorical variables
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X = pd.get_dummies(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save the model
save_model(model, 'models/diagnosis_model.pkl')
print("Model trained and saved to models/diagnosis_model.pkl")
