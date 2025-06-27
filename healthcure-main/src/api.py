from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load your trained model
try:
    model = joblib.load('models/diagnosis_model.pkl')
    # If your model was trained with specific columns, get them for validation
    feature_names = getattr(model, 'feature_names_in_', None)
except Exception as e:
    print("Failed to load model:", e)
    model = None
    feature_names = None

@app.route('/predict', methods=['POST'])
def predict():
    # Return error if model didn't load
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        print("Received data:", data)

        # If feature_names are known, reindex for safety
        if feature_names is not None:
            df = pd.DataFrame([data], columns=feature_names)
        else:
            df = pd.DataFrame([data])

        print("DataFrame for prediction:\n", df)
        prediction = model.predict(df)
        print("Prediction:", prediction)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print("Error during prediction:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return "HealthCure API is running."

if __name__ == '__main__':
    app.run(debug=True)