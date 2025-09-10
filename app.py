from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from datetime import datetime
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model and preprocessing objects
try:
    model = joblib.load('traffic_prediction_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    logging.info("Model and preprocessing objects loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None

# Mapping original labels to frontend labels
PREDICTION_MAP = {
    "heavy traffic": "heavy",
    "heavy": "heavy",
    "high traffic": "high",
    "high": "high",
    "normal traffic": "normal",
    "normal": "normal",
    "low traffic": "low",
    "low": "low"
}


def normalize_label(label: str) -> str:
    """Normalize prediction labels to match frontend expectations"""
    return PREDICTION_MAP.get(label.lower(), label.lower())


def predict_traffic(car_count, bike_count, bus_count, truck_count, hour, minute, day_of_week, is_weekend):
    """
    Predict traffic situation based on input parameters
    """
    if model is None:
        return "Model not loaded", {}

    try:
        # Encode categorical variables
        day_encoded = label_encoders['Day of the week'].transform([day_of_week])[0]

        # Determine time of day
        if 0 <= hour < 6:
            time_of_day = 'Night'
        elif 6 <= hour < 12:
            time_of_day = 'Morning'
        elif 12 <= hour < 18:
            time_of_day = 'Afternoon'
        else:
            time_of_day = 'Evening'

        time_encoded = label_encoders['Time_of_Day'].transform([time_of_day])[0]

        # Create feature array
        features_array = np.array([[car_count, bike_count, bus_count, truck_count,
                                   hour, minute, is_weekend, day_encoded, time_encoded]])

        logging.info(f"Features before scaling: {features_array}")

        # Apply scaling if used during training
        try:
            features_array = scaler.transform(features_array)
            logging.info(f"Features after scaling: {features_array}")
        except Exception as e:
            logging.warning(f"Scaler not applied: {str(e)}")

        # Make prediction
        prediction_encoded = model.predict(features_array)[0]
        prediction_raw = label_encoders['Traffic Situation'].inverse_transform([prediction_encoded])[0]
        prediction = normalize_label(prediction_raw)

        # Get prediction probabilities
        probabilities = model.predict_proba(features_array)[0]

        # Normalize probability dictionary
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            class_name = label_encoders['Traffic Situation'].classes_[i]
            prob_dict[normalize_label(class_name)] = round(prob * 100, 2)

        logging.info(f"Prediction raw: {prediction_raw}, normalized: {prediction}")
        logging.info(f"Probabilities: {prob_dict}")

        return prediction, prob_dict
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return "Error in prediction", {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        car_count = int(request.form['car_count'])
        bike_count = int(request.form['bike_count'])
        bus_count = int(request.form['bus_count'])
        truck_count = int(request.form['truck_count'])

        # Get time and date information
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        day_of_week = request.form['day_of_week']
        is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0

        # Make prediction
        prediction, probabilities = predict_traffic(
            car_count, bike_count, bus_count, truck_count,
            hour, minute, day_of_week, is_weekend
        )

        # Map prediction to alert type + icon
        if prediction == "heavy":
            alert_type = "danger"
            icon = "⚠️"
        elif prediction == "high":
            alert_type = "warning"
            icon = "🔶"
        elif prediction == "normal":
            alert_type = "info"
            icon = "ℹ️"
        else:  # low
            alert_type = "success"
            icon = "✅"

        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'icon': icon,
            'alert_type': alert_type,
            'probabilities': probabilities,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in prediction route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
