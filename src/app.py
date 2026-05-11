import streamlit as st # type:ignore
import joblib
import pandas as pd
import numpy as np

# 1. Load our tools
model = joblib.load('src/housing_model.pkl')
model_columns = joblib.load('src/model_columns.pkl')

st.title("🏡 AI House Price Predictor")
st.write("Adjust the details below to see the estimated market price.")

# 2. Create the input fields (The Sliders)
# We'll pick a few high-impact features to start
area = st.slider("Total Living Area (sq ft)", 500, 5000, 1500)
quality = st.slider("Overall Quality (1-10)", 1, 10, 5)
year = st.slider("Year Built", 1900, 2026, 2000)

# 3. Predict Button
if st.button("Predict Price"):
    # Prepare the data exactly like we did for the model
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Map our sliders to the correct columns
    input_data['GrLivArea'] = area
    input_data['OverallQual'] = quality
    input_data['YearBuilt'] = year
    
    # The Math
    pred_log = model.predict(input_data)
    final_price = np.expm1(pred_log)[0]
    
    st.success(f"### Estimated Price: ${final_price:,.2f}")