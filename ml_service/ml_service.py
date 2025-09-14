from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js communication

# Load the trained model once when server starts
try:
    model_data = joblib.load('aquaintel_model.pkl')
    model = model_data['model']
    features = model_data['features']
    model_name = model_data['model_name']
    metrics = model_data['metrics']
    print(f"Model loaded: {model_name} (R²={metrics['R2']:.3f})")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_crisis(model, current_features, critical=2.0, horizon=45):
    """Predict water crisis timeline"""
    preds = []
    curr_df = pd.DataFrame([current_features], columns=features)
    
    for day in range(horizon):
        p = model.predict(curr_df)[0]
        preds.append(float(p))  # Convert to Python float
        curr_df.at[0, 'WL_lag_1'] = p
        
        if p <= critical:
            return day + 1, preds
    
    return None, preds

def format_message(days, loc="Location"):
    """Format crisis message"""
    if days is None:
        return f"{loc} water stable for next 45 days."
    if days <= 7:
        return f"URGENT: {loc} has {days} days of water left!"
    if days <= 15:
        return f"WARNING: {loc} has {days} days left."
    return f"{loc} has {days} days of water left."

def get_alert_level(days):
    """Determine alert level"""
    if days is None:
        return "LOW"
    if days <= 7:
        return "CRITICAL"
    if days <= 15:
        return "HIGH"
    if days <= 30:
        return "MEDIUM"
    return "LOW"

def get_recommendations(days):
    """Generate recommendations based on crisis timeline"""
    if days is None:
        return [
            "Continue regular monitoring",
            "Maintain current water usage patterns",
            "Monitor seasonal changes"
        ]
    elif days <= 7:
        return [
            "Implement emergency water conservation immediately",
            "Contact local water authorities urgently",
            "Seek alternative water sources",
            "Restrict all non-essential water usage"
        ]
    elif days <= 15:
        return [
            "Implement water conservation measures",
            "Alert local water authorities for potential shortages",
            "Consider using alternative water sources if available",
            "Reduce non-essential water usage"
        ]
    elif days <= 30:
        return [
            "Encourage voluntary water conservation",
            "Check and repair any known leaks",
            "Plan for potential seasonal shortages",
            "Increase monitoring frequency"
        ]
    else:
        return [
            "Continue regular monitoring practices",
            "Maintain current water usage patterns",
            "Promote sustainable water management"
        ]

@app.route('/predict', methods=['POST'])
def predict():
    """ML Prediction endpoint"""
    if model is None:
        return jsonify({
            'error': 'ML model not loaded'
        }), 500
    
    try:
        data = request.json
        
        # Extract input parameters
        village_name = data.get('village_name', 'Unknown Location')
        temp = float(data.get('temperature', 25.0))
        rain = float(data.get('rainfall', 0.0))
        ph = float(data.get('ph', 7.0))
        do = float(data.get('dissolved_oxygen', 7.0))
        current_wl = float(data.get('current_water_level', 3.0))
        
        # Build feature vector
        now = datetime.datetime.now()
        curr = {
            'Temperature_C': temp,
            'Rainfall_mm': rain,
            'pH': ph,
            'Dissolved_Oxygen_mg_L': do,
            'day_of_year': now.timetuple().tm_yday,
            'month': now.month,
            'WL_lag_1': current_wl,
            'WL_lag_3': current_wl,
            'WL_lag_7': current_wl,
            'WL_lag_14': current_wl,
            'WL_lag_30': current_wl,
            'WL_roll_mean_7': current_wl,
            'WL_roll_mean_14': current_wl,
            'WL_roll_mean_30': current_wl,
            'WL_roll_std_7': 0,
            'WL_roll_std_14': 0,
            'WL_roll_std_30': 0
        }
        
        # Make prediction
        days, forecast = predict_crisis(model, curr)
        msg = format_message(days, village_name)
        alert_level = get_alert_level(days)
        recommendations = get_recommendations(days)
        
        # Return prediction result
        result = {
            'village_name': village_name,
            'days_until_crisis': days,
            'crisis_message': msg,
            'alert_level': alert_level,
            'confidence_score': float(metrics['R2']),
            'recommendations': recommendations,
            'forecast_7_days': forecast[:7] if forecast else [],
            'model_info': {
                'type': model_name,
                'accuracy': f"{metrics['R2']:.2%}",
                'mae': f"{metrics['MAE']:.3f}"
            },
            'current_water_level': current_wl,
            'input_parameters': {
                'temperature': temp,
                'rainfall': rain,
                'ph': ph,
                'dissolved_oxygen': do
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_name if model else None
    })

if __name__ == '__main__':
    print("Starting AquaIntel ML Service...")
    app.run(host='0.0.0.0', port=5000, debug=True)
