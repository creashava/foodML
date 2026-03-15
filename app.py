from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
le_day = joblib.load('le_day.pkl')
le_weather = joblib.load('le_weather.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        day = request.form['day']
        weather = request.form['weather']
        festival = int(request.form['festival'])
        expected_customers = int(request.form['expected_customers'])
        prev_day = int(request.form['prev_day'])
        prev_week = int(request.form['prev_week'])

        if not (1 <= expected_customers <= 1000):
            raise ValueError("Expected customers must be between 1 and 1000")
        if not (1 <= prev_day <= 1000):
            raise ValueError("Previous day consumption must be between 1 and 1000")
        if not (1 <= prev_week <= 1000):
            raise ValueError("Previous week consumption must be between 1 and 1000")

        day_encoded = le_day.transform([day])[0]
        weather_encoded = le_weather.transform([weather])[0]
        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
        demand_avg = (prev_day + prev_week) / 2

        features = np.array([[day_encoded, festival, weather_encoded,
                              expected_customers, prev_day,
                              prev_week, is_weekend, demand_avg]])

        prediction = model.predict(features)[0]
        return render_template('index.html', prediction=int(prediction))

    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error="Something went wrong. Please check your inputs.")


if __name__ == '__main__':
    app.run(debug=True)