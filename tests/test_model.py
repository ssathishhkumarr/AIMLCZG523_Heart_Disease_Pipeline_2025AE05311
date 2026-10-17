import joblib
import pandas as pd
import os

def test_model_exists():
    # Verify the model was successfully saved during training
    assert os.path.exists("models/model.pkl"), "Model file does not exist in the models/ directory"

def test_model_prediction():
    # Load the packaged model
    model = joblib.load("models/model.pkl")
    
    # Create a dummy patient record with the 13 required features
    dummy_patient = pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0.0, "thal": 6.0
    }])
    
    # Generate a prediction
    prediction = model.predict(dummy_patient)
    
    # Assert the output is a valid binary classification (0 or 1)
    assert prediction[0] in [0, 1], "Model prediction must be binary (0 or 1)"