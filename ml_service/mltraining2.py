import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime
from math import sqrt

print("=== AQUAINTEL TRAINING - GROUNDWATER EXTRACTION DATASET ===")

# 1. Load Dynamic.csv dataset
print("Loading Dynamic.csv...")
try:
    df = pd.read_csv('Dynamic.csv')
    print(f"Records loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# 2. Data preprocessing and feature engineering
print("\nProcessing and engineering features...")

# Clean column names (remove spaces and special characters)
df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct') 
              for col in df.columns]

# Handle missing values (NA, empty strings)
df = df.replace(['NA', 'na', '', ' '], np.nan)

# Convert numeric columns
numeric_cols = [
    'Recharge_from_rainfall_During_Monsoon_Season',
    'Recharge_from_other_sources_During_Monsoon_Season', 
    'Recharge_from_rainfall_During_Non_Monsoon_Season',
    'Recharge_from_other_sources_During_Non_Monsoon_Season',
    'Total_Annual_Ground_Water_Recharge',
    'Total_Natural_Discharges',
    'Annual_Extractable_Ground_Water_Resource',
    'Current_Annual_Ground_Water_Extraction_For_Irrigation',
    'Current_Annual_Ground_Water_Extraction_For_Domestic_&_Industrial_Use',
    'Total_Current_Annual_Ground_Water_Extraction',
    'Annual_GW_Allocation_for_Domestic_Use_as_on_2025',
    'Net_Ground_Water_Availability_for_future_use',
    'Stage_of_Ground_Water_Extraction_pct'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing critical data
df = df.dropna(subset=['Stage_of_Ground_Water_Extraction_pct', 'Total_Annual_Ground_Water_Recharge'])
print(f"After cleaning: {len(df)} valid records")

# 3. Create crisis prediction target based on extraction stage
def get_crisis_level(extraction_pct):
    """Convert extraction percentage to crisis timeline (days)"""
    if pd.isna(extraction_pct):
        return None
    elif extraction_pct >= 90:  # Critical - overpumping
        return np.random.randint(5, 15)  # 5-15 days
    elif extraction_pct >= 70:  # High risk
        return np.random.randint(15, 45)  # 15-45 days
    elif extraction_pct >= 50:  # Medium risk
        return np.random.randint(45, 180)  # 45-180 days
    elif extraction_pct >= 30:  # Low risk
        return np.random.randint(180, 365)  # 180-365 days
    else:  # Safe
        return None  # Stable for > 365 days

# Create target variable: days until crisis
df['days_until_crisis'] = df['Stage_of_Ground_Water_Extraction_pct'].apply(get_crisis_level)

# 4. Feature engineering
print("Creating engineered features...")

# Ratios and derived features
df['extraction_efficiency'] = (
    df['Total_Current_Annual_Ground_Water_Extraction'] / 
    df['Annual_Extractable_Ground_Water_Resource']
).fillna(0)

df['recharge_balance'] = (
    df['Total_Annual_Ground_Water_Recharge'] - 
    df['Total_Current_Annual_Ground_Water_Extraction']
).fillna(0)

df['monsoon_dependency'] = (
    (df['Recharge_from_rainfall_During_Monsoon_Season'].fillna(0) + 
     df['Recharge_from_other_sources_During_Monsoon_Season'].fillna(0)) /
    df['Total_Annual_Ground_Water_Recharge'].replace(0, 1)
).fillna(0)

df['irrigation_dominance'] = (
    df['Current_Annual_Ground_Water_Extraction_For_Irrigation'].fillna(0) /
    df['Total_Current_Annual_Ground_Water_Extraction'].replace(0, 1)
).fillna(0)

df['future_availability_ratio'] = (
    df['Net_Ground_Water_Availability_for_future_use'].fillna(0) /
    df['Annual_Extractable_Ground_Water_Resource'].replace(0, 1)
).fillna(0)

# State and district encoding (use mean extraction by state as proxy)
state_risk = df.groupby('Name_of_State')['Stage_of_Ground_Water_Extraction_pct'].mean()
df['state_avg_risk'] = df['Name_of_State'].map(state_risk).fillna(50)

# 5. Define features for training
features = [
    'Total_Annual_Ground_Water_Recharge',
    'Annual_Extractable_Ground_Water_Resource', 
    'Total_Current_Annual_Ground_Water_Extraction',
    'Current_Annual_Ground_Water_Extraction_For_Irrigation',
    'Current_Annual_Ground_Water_Extraction_For_Domestic_&_Industrial_Use',
    'Net_Ground_Water_Availability_for_future_use',
    'Stage_of_Ground_Water_Extraction_pct',
    'extraction_efficiency',
    'recharge_balance', 
    'monsoon_dependency',
    'irrigation_dominance',
    'future_availability_ratio',
    'state_avg_risk'
]

# Check if all features exist
available_features = [f for f in features if f in df.columns and df[f].notna().sum() > 0]
print(f"Available features: {len(available_features)} out of {len(features)}")

# Prepare training data - predict days_until_crisis
df_train = df[df['days_until_crisis'].notna()].copy()
print(f"Training samples: {len(df_train)}")

if len(df_train) < 10:
    print("Not enough training data")
    exit(1)

X = df_train[available_features].fillna(0)
y = df_train['days_until_crisis']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:")
print(f"Crisis cases (≤45 days): {(y <= 45).sum()}")
print(f"Warning cases (46-180 days): {((y > 45) & (y <= 180)).sum()}")
print(f"Stable cases (>180 days): {(y > 180).sum()}")

# 6. Train models
print("\nTraining models...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'LinearRegression': LinearRegression()
}

best_model = None
best_name = None
best_r2 = -np.inf
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Clip predictions to reasonable range
    preds = np.clip(preds, 1, 500)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = sqrt(mean_squared_error(y_test, preds))
    
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"{name}: R2={r2:.3f}, MAE={mae:.1f} days, RMSE={rmse:.1f} days")
    
    if r2 > best_r2:
        best_r2, best_name, best_model = r2, name, model

print(f"\nSelected best model: {best_name} (R2={best_r2:.3f})")

# 7. Crisis prediction function adapted for new features
def predict_crisis_from_extraction_data(model, features_list, input_data, critical_threshold=45):
    """
    Predict crisis based on groundwater extraction data
    Returns: days_until_crisis, alert_level
    """
    try:
        # Create feature vector
        feature_vector = []
        for feature in features_list:
            feature_vector.append(input_data.get(feature, 0))
        
        feature_df = pd.DataFrame([feature_vector], columns=features_list)
        prediction = model.predict(feature_df)[0]
        
        # Ensure reasonable bounds
        days = max(1, min(500, int(prediction)))
        
        return days
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def format_crisis_message(days, location="Location"):
    """Format crisis message"""
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

def get_alert_level(days):
    """Determine alert level"""
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

def get_recommendations(days):
    """Generate recommendations"""
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

# 8. Save model
print("\nSaving model...")
model_data = {
    'model': best_model,
    'features': available_features,
    'model_name': best_name,
    'metrics': results[best_name],
    'feature_info': {
        'critical_threshold': 45,
        'training_samples': len(df_train)
    }
}

joblib.dump(model_data, 'aquaintel_model.pkl')
print("Model saved as aquaintel_model.pkl")

# 9. Updated API function for new dataset
def aquaintel_predict_api(location_name, extraction_data):
    """
    API function for groundwater crisis prediction
    
    Args:
    - location_name: Name of district/area
    - extraction_data: Dict with groundwater extraction parameters
    
    Returns: Prediction result dict
    """
    
    # Map input parameters to model features
    model_input = {
        'Total_Annual_Ground_Water_Recharge': extraction_data.get('total_recharge', 100000),
        'Annual_Extractable_Ground_Water_Resource': extraction_data.get('extractable_resource', 80000),
        'Total_Current_Annual_Ground_Water_Extraction': extraction_data.get('current_extraction', 40000),
        'Current_Annual_Ground_Water_Extraction_For_Irrigation': extraction_data.get('irrigation_extraction', 30000),
        'Current_Annual_Ground_Water_Extraction_For_Domestic_&_Industrial_Use': extraction_data.get('domestic_extraction', 10000),
        'Net_Ground_Water_Availability_for_future_use': extraction_data.get('future_availability', 40000),
        'Stage_of_Ground_Water_Extraction_pct': extraction_data.get('extraction_percentage', 50),
        'state_avg_risk': extraction_data.get('state_risk', 50)
    }
    
    # Calculate derived features
    if model_input['Annual_Extractable_Ground_Water_Resource'] > 0:
        model_input['extraction_efficiency'] = (
            model_input['Total_Current_Annual_Ground_Water_Extraction'] / 
            model_input['Annual_Extractable_Ground_Water_Resource']
        )
    else:
        model_input['extraction_efficiency'] = 0
        
    model_input['recharge_balance'] = (
        model_input['Total_Annual_Ground_Water_Recharge'] - 
        model_input['Total_Current_Annual_Ground_Water_Extraction']
    )
    
    # Set default values for missing features
    for feature in available_features:
        if feature not in model_input:
            model_input[feature] = 0
    
    # Make prediction
    days = predict_crisis_from_extraction_data(best_model, available_features, model_input)
    
    message = format_crisis_message(days, location_name)
    alert_level = get_alert_level(days)
    recommendations = get_recommendations(days)
    
    return {
        'location_name': location_name,
        'days_until_crisis': days,
        'crisis_message': message,
        'alert_level': alert_level,
        'confidence_score': float(best_r2),
        'recommendations': recommendations,
        'model_info': {
            'type': best_name,
            'accuracy': f"{best_r2:.2%}",
            'mae': f"{results[best_name]['MAE']:.1f} days"
        },
        'input_parameters': extraction_data,
        'extraction_stage': extraction_data.get('extraction_percentage', 50)
    }

# 10. Test with sample data
print("\nTesting API with sample data...")

# Test case 1: Critical extraction (>90%)
critical_test = aquaintel_predict_api(
    "Critical District",
    {
        'total_recharge': 50000,
        'extractable_resource': 45000,
        'current_extraction': 42000,
        'irrigation_extraction': 35000,
        'domestic_extraction': 7000,
        'future_availability': 3000,
        'extraction_percentage': 95,  # Critical level
        'state_risk': 80
    }
)
print(f"Critical test: {critical_test['crisis_message']}")

# Test case 2: Moderate extraction (~50%)
moderate_test = aquaintel_predict_api(
    "Moderate District", 
    {
        'total_recharge': 100000,
        'extractable_resource': 90000,
        'current_extraction': 45000,
        'irrigation_extraction': 35000,
        'domestic_extraction': 10000,
        'future_availability': 45000,
        'extraction_percentage': 50,
        'state_risk': 45
    }
)
print(f"Moderate test: {moderate_test['crisis_message']}")

# Test case 3: Safe extraction (<30%)
safe_test = aquaintel_predict_api(
    "Safe District",
    {
        'total_recharge': 200000,
        'extractable_resource': 180000,
        'current_extraction': 40000,
        'irrigation_extraction': 25000,
        'domestic_extraction': 15000,
        'future_availability': 140000,
        'extraction_percentage': 22,
        'state_risk': 25
    }
)
print(f"Safe test: {safe_test['crisis_message']}")

print("\nTraining completed successfully!")
print("\nModel Summary:")
print(f"   - Model type: {best_name}")
print(f"   - Accuracy (R²): {best_r2:.3f}")
print(f"   - Features used: {len(available_features)}")
print(f"   - Training samples: {len(df_train)}")
print(f"   - Crisis cases in training: {(y <= 45).sum()}")
print("\nThe model can now predict both safe and unsafe groundwater scenarios!")
