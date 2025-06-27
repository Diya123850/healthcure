from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data and print a classification report and accuracy.

    Args:
        model: Trained scikit-learn model.
        X_test: Test features.
        y_test: True labels for the test set.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    import pandas as pd
    import joblib

    # Load test data (update the path and preprocessing as per your pipeline)
    data = pd.read_csv('data/sample_patient_data.csv')
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    X = pd.get_dummies(X)

    # You may want to use the same split as during training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the model
    model = joblib.load('models/diagnosis_model.pkl')

    # Evaluate
    evaluate_model(model, X_test, y_test)
