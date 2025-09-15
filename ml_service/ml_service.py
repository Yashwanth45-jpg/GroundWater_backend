from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import datetime
from flask_cors import CORS
from typing import Dict, Any, Optional, List

app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js communication

# Global variables for model data
model = None
features = []
model_name = ""
metrics = {}
feature_info = {}

# Load the trained model once when server starts
try:
    model_data = joblib.load('aquaintel_model.pkl')
    model = model_data['model']
    features = model_data['features']
    model_name = model_data['model_name']
    metrics = model_data['metrics']
    feature_info = model_data.get('feature_info', {})
    print(f"Model loaded: {model_name} (R²={metrics['R2']:.3f})")
    print(f"Features: {len(features)}")
    print(f"Training samples: {feature_info.get('training_samples', 'Unknown')}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_crisis_from_extraction_data(model, features_list: List[str], input_data: Dict[str, float]) -> Optional[int]:
    """
    Predict crisis based on groundwater extraction data
    Returns: days_until_crisis
    """
    try:
        # Create feature vector in the same order as training
        feature_vector = []
        for feature in features_list:
            feature_vector.append(input_data.get(feature, 0))
        
        # Create DataFrame for prediction
        feature_df = pd.DataFrame([feature_vector], columns=features_list)
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Ensure reasonable bounds (1-500 days)
        days = max(1, min(500, int(prediction)))
        
        return days
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def format_crisis_message(days: Optional[int], location: str = "Location") -> str:
    """Format crisis message based on days until crisis"""
    if days is None:
        return f"Unable to predict for {location}"
    elif days <= 15:
        return f"CRITICAL: {location} - severe water stress in {days} days!"
    elif days <= 45:
        return f"HIGH RISK: {location} - water crisis expected in {days} days"
    elif days <= 180:
        return f"MEDIUM RISK: {location} - potential issues in {days} days"
    else:
        return f"LOW RISK: {location} - stable for {days} days"

def get_alert_level(days: Optional[int]) -> str:
    """Determine alert level based on days until crisis"""
    if days is None:
        return "UNKNOWN"
    elif days <= 15:
        return "CRITICAL"
    elif days <= 45:
        return "HIGH"
    elif days <= 180:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommendations(days: Optional[int]) -> List[str]:
    """Generate recommendations based on crisis timeline"""
    if days is None:
        return ["Unable to generate recommendations - insufficient data"]
    elif days <= 15:
        return [
            "IMMEDIATE ACTION: Stop all non-essential water extraction",
            "Alert state water board and district collector",
            "Implement emergency water rationing",
            "Deploy water tankers for critical needs",
            "Ban new tube wells and bore wells"
        ]
    elif days <= 45:
        return [
            "Implement strict water conservation measures",
            "Reduce industrial water allocation by 30%",
            "Switch to drought-resistant crop varieties", 
            "Increase rainwater harvesting infrastructure",
            "Launch public awareness campaign"
        ]
    elif days <= 180:
        return [
            "Monitor groundwater levels weekly",
            "Accelerate recharge projects during monsoon",
            "Promote micro-irrigation systems",
            "Review and optimize water allocation",
            "Coordinate with neighboring districts"
        ]
    else:
        return [
            "Continue sustainable extraction practices",
            "Maintain regular monitoring schedule",
            "Promote water-efficient agriculture",
            "Plan for future water infrastructure"
        ]

def calculate_derived_features(extraction_data: Dict[str, float]) -> Dict[str, float]:
    """Calculate derived features from raw input data"""
    derived = {}
    
    # Basic values with defaults
    total_recharge = extraction_data.get('total_recharge', 100000)
    extractable_resource = extraction_data.get('extractable_resource', 80000)
    current_extraction = extraction_data.get('current_extraction', 40000)
    
    # Calculate extraction efficiency
    if extractable_resource > 0:
        derived['extraction_efficiency'] = current_extraction / extractable_resource
    else:
        derived['extraction_efficiency'] = 0
    
    # Calculate recharge balance
    derived['recharge_balance'] = total_recharge - current_extraction
    
    # Monsoon dependency (simplified - assume 70% from monsoon)
    derived['monsoon_dependency'] = 0.7
    
    # Irrigation dominance
    irrigation_extraction = extraction_data.get('irrigation_extraction', current_extraction * 0.75)
    if current_extraction > 0:
        derived['irrigation_dominance'] = irrigation_extraction / current_extraction
    else:
        derived['irrigation_dominance'] = 0.75
    
    # Future availability ratio
    future_availability = extraction_data.get('future_availability', 40000)
    if extractable_resource > 0:
        derived['future_availability_ratio'] = future_availability / extractable_resource
    else:
        derived['future_availability_ratio'] = 0.5
    
    return derived

@app.route('/predict', methods=['POST'])
def predict():
    """ML Prediction endpoint for groundwater extraction data"""
    if model is None:
        return jsonify({
            'error': 'ML model not loaded'
        }), 500
    
    try:
        # Get request data and check if it exists
        data = request.get_json()
        if data is None:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
            
        print(f"Received prediction request: {data.get('village_name', 'Unknown')}")
        
        # Extract input parameters - now focused on groundwater extraction data
        village_name = data.get('village_name', 'Unknown Location')
        
        # Primary extraction parameters
        extraction_data = {
            'total_recharge': float(data.get('total_recharge', 100000)),
            'extractable_resource': float(data.get('extractable_resource', 80000)),
            'current_extraction': float(data.get('current_extraction', 40000)),
            'irrigation_extraction': float(data.get('irrigation_extraction', 30000)),
            'domestic_extraction': float(data.get('domestic_extraction', 10000)),
            'future_availability': float(data.get('future_availability', 40000)),
            'extraction_percentage': float(data.get('extraction_percentage', 50)),
            'state_risk': float(data.get('state_risk', 50))
        }
        
        # Backward compatibility - if old parameters are provided, convert them
        # Check if data exists and has the keys before using 'in' operator
        has_legacy_params = False
        if data and isinstance(data, dict):
            has_legacy_params = 'temperature' in data or 'current_water_level' in data
        
        if has_legacy_params:
            print("Converting legacy parameters to extraction data...")
            # Use defaults based on typical Indian groundwater data
            current_wl = float(data.get('current_water_level', 3.0))
            
            # Rough conversion: lower water level = higher extraction stress
            if current_wl <= 1.5:
                extraction_data['extraction_percentage'] = 95  # Critical
            elif current_wl <= 2.5:
                extraction_data['extraction_percentage'] = 75  # High
            elif current_wl <= 4.0:
                extraction_data['extraction_percentage'] = 50  # Medium
            else:
                extraction_data['extraction_percentage'] = 25  # Low
        
        # Build model input features
        model_input = {
            'Total_Annual_Ground_Water_Recharge': extraction_data['total_recharge'],
            'Annual_Extractable_Ground_Water_Resource': extraction_data['extractable_resource'],
            'Total_Current_Annual_Ground_Water_Extraction': extraction_data['current_extraction'],
            'Current_Annual_Ground_Water_Extraction_For_Irrigation': extraction_data['irrigation_extraction'],
            'Current_Annual_Ground_Water_Extraction_For_Domestic_&_Industrial_Use': extraction_data['domestic_extraction'],
            'Net_Ground_Water_Availability_for_future_use': extraction_data['future_availability'],
            'Stage_of_Ground_Water_Extraction_pct': extraction_data['extraction_percentage'],
            'state_avg_risk': extraction_data['state_risk']
        }
        
        # Add derived features
        derived_features = calculate_derived_features(extraction_data)
        model_input.update(derived_features)
        
        # Set default values for any missing features
        for feature in features:
            if feature not in model_input:
                model_input[feature] = 0
                
        print(f"Making prediction for {village_name} (extraction: {extraction_data['extraction_percentage']:.1f}%)")
        
        # Make prediction
        days = predict_crisis_from_extraction_data(model, features, model_input)
        
        if days is None:
            return jsonify({
                'error': 'Prediction failed - model returned invalid result'
            }), 500
        
        # Format response
        msg = format_crisis_message(days, village_name)
        alert_level = get_alert_level(days)
        recommendations = get_recommendations(days)
        
        # Generate forecast (simplified - gradual decline based on extraction rate)
        forecast_7_days = []
        if days <= 180:  # Only generate forecast for at-risk areas
            daily_decline_rate = extraction_data['extraction_percentage'] / 365.0 / 100.0
            base_level = 10.0 - (extraction_data['extraction_percentage'] / 10.0)  # Rough water level estimate
            for i in range(7):
                forecast_7_days.append(max(0.1, base_level - (daily_decline_rate * i)))
        else:
            # Stable areas - minimal change
            base_level = 5.0
            for i in range(7):
                forecast_7_days.append(base_level - (i * 0.01))
        
        # Prepare result
        result = {
            'village_name': village_name,
            'days_until_crisis': days,
            'crisis_message': msg,
            'alert_level': alert_level,
            'confidence_score': float(metrics.get('R2', 0.0)),
            'recommendations': recommendations,
            'forecast_7_days': forecast_7_days,
            'model_info': {
                'type': model_name,
                'accuracy': f"{metrics.get('R2', 0.0):.2%}",
                'mae': f"{metrics.get('MAE', 0.0):.1f} days"
            },
            'extraction_stage': extraction_data['extraction_percentage'],
            'input_parameters': {
                'total_recharge': extraction_data['total_recharge'],
                'extractable_resource': extraction_data['extractable_resource'],
                'current_extraction': extraction_data['current_extraction'],
                'extraction_percentage': extraction_data['extraction_percentage'],
                'future_availability': extraction_data['future_availability']
            }
        }
        
        print(f"Prediction successful: {alert_level} - {days} days")
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input data: {str(e)}'
        }), 400
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict-legacy', methods=['POST'])
def predict_legacy():
    """Legacy endpoint for backward compatibility with old sensor data format"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        village_name = data.get('village_name', 'Unknown Location')
        current_wl = float(data.get('current_water_level', 3.0))
        
        # Convert water level to extraction data
        if current_wl <= 1.5:
            extraction_pct = 95
        elif current_wl <= 2.5:
            extraction_pct = 75
        elif current_wl <= 4.0:
            extraction_pct = 50
        else:
            extraction_pct = 25
            
        # Create converted data for main prediction
        converted_data = {
            'village_name': village_name,
            'extraction_percentage': extraction_pct,
            'total_recharge': 100000,
            'extractable_resource': 80000,
            'current_extraction': int(80000 * extraction_pct / 100)
        }
        
        # Simulate internal request
        with app.test_request_context(json=converted_data, method='POST', content_type='application/json'):
            return predict()
            
    except Exception as e:
        return jsonify({
            'error': f'Legacy prediction failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_name if model else None,
        'features_count': len(features) if features else 0,
        'dataset_type': 'groundwater_extraction'
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Return list of model features for debugging"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    return jsonify({
        'features': features,
        'feature_count': len(features),
        'model_type': model_name,
        'sample_input': {
            'village_name': 'Sample District',
            'total_recharge': 100000,
            'extractable_resource': 80000,
            'current_extraction': 40000,
            'irrigation_extraction': 30000,
            'domestic_extraction': 10000,
            'future_availability': 40000,
            'extraction_percentage': 50,
            'state_risk': 45
        }
    })

if __name__ == '__main__':
    print("Starting AquaIntel ML Service (Groundwater Extraction Model)...")
    print("New model focuses on groundwater extraction patterns")
    print("Can now predict both safe and critical water scenarios")
    app.run(host='0.0.0.0', port=5000, debug=True)
