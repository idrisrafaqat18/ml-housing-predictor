import joblib
import pandas as pd
import numpy as np

def predict_house_price(input_data):
    # 1. Load the "Frozen Brain"
    model = joblib.load('src/housing_model.pkl')
    
    # 2. Convert input to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 3. Handle the 'Log' math (The model predicts in Log, we want Dollars)
    prediction_log = model.predict(input_df)
    final_price = np.expm1(prediction_log)
    
    return final_price[0]

# --- Testing it out ---
# Let's pretend we have a new house (simplified example)
# Note: In a real app, this dictionary would need ALL 224 columns
print("AI is calculating price...")
# Example dummy data - this is just a placeholder for now
# actual_prediction = predict_house_price(some_data)