import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Load preprocessed data (should have no missing values)
    data = pd.read_csv('data/preprocessed_patient_data.csv')
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save scaled features
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/X_test_scaled.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    print("Feature engineering complete. Encoded and scaled data saved.")