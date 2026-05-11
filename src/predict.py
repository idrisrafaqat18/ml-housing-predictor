import joblib
import pandas as pd
import numpy as np

def predict():
    # 1. Load the Model and the Column Map
    model = joblib.load('src/housing_model.pkl')
    model_columns = joblib.load('src/model_columns.pkl')
    
    # 2. Create a "Dummy House" with all zeros for all 224 columns
    # In a real app, this data would come from a website form
    sample_house = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 3. Fill in some details for our "Test House"
    # (Using some common column names from the dataset)
    if 'GrLivArea' in sample_house.columns:
        sample_house['GrLivArea'] = 2000  # 2000 square feet
    if 'OverallQual' in sample_house.columns:
        sample_house['OverallQual'] = 7   # 7/10 Quality
        
    # 4. Predict
    prediction_log = model.predict(sample_house)
    final_price = np.expm1(prediction_log)
    
    print(f"🏠 Estimated House Price: ${final_price[0]:,.2f}")

if __name__ == "__main__":
    predict()