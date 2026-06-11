from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load assets
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    model, scaler, metrics = None, None, {"MSE": 0, "RMSE": 0, "R2": 0}
    print("Warning: Model files not found. Please run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model is not trained yet."})

    try:
        data = request.json
        
        # Extract fields from form
        area = float(data.get('area', 1500))
        bedrooms = float(data.get('bedrooms', 3))
        bathrooms = float(data.get('bathrooms', 2))
        location_score = float(data.get('location_score', 8.5))

        # Map to California Housing Features
        # MedInc = Location score * 0.5 (Scale 1-10 mapped to approx 0.5 - 5.0)
        med_inc = location_score * 0.5
        # HouseAge = Set to California average
        house_age = 28.0
        # AveRooms = Area / 300 (rough estimate)
        ave_rooms = area / 300.0
        # AveBedrms = Directly from input
        ave_bedrms = bedrooms
        # Population = Average block population
        population = 1425.0
        # AveOccup = Average occupancy
        ave_occup = 3.0
        # Latitude = California avg
        latitude = 35.6
        # Longitude = California avg
        longitude = -119.5

        # Create DataFrame
        input_data = pd.DataFrame([[
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude
        ]], columns=[
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ])

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        
        # Convert to $ value
        price = prediction * 100000

        # Return formatted USD
        return jsonify({
            "price": f"${price:,.0f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
