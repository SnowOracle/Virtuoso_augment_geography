# Model Directory

This directory stores trained machine learning models for the recommendation system.

## Files

- `recommendation_model.pkl` - XGBoost model for destination recommendations
- `encoders.pkl` - Serialized OneHotEncoder objects for categorical features
- `scaler.pkl` - StandardScaler for numerical features

## Usage

Models are automatically loaded by the `recommendation_system.py` module when the application starts. If no models exist, the system will train new models when recommendations are requested.

## Regenerating Models

To force model retraining:

1. Delete the model files in this directory
2. Start the application
3. Go to the Recommendations tab
4. Click "Generate Recommendations"

The system will automatically train new models based on the current data. 