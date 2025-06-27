from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        patient_data = request.form.to_dict()
        response = requests.post('http://localhost:5000/predict', json=patient_data)
        prediction = response.json().get('prediction')
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=8080, debug=True)