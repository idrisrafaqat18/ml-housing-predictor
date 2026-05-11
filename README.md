# 🏡 ML Housing Price Predictor

An end-to-end Machine Learning pipeline that predicts house prices with **88% accuracy** using advanced regression techniques. This project transitions from raw data exploration to a functional web application.

## 🚀 Project Overview
This project uses the Ames Housing Dataset to predict residential home prices. I implemented a full ML lifecycle, including data cleaning, feature engineering, and model deployment.

### Key Performance:
* **Baseline Model:** Linear Regression (R²: 0.75)
* **Champion Model:** Random Forest Regressor (R²: 0.88)
* **Mean Absolute Error:** ~$17,724

## 🛠️ The Tech Stack
* **Python 3.13** (Core Logic)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Machine Learning)
* **Streamlit** (Web Interface)
* **Matplotlib & Seaborn** (Data Visualization)

## 🧠 Engineering Highlights
1. **Feature Engineering:** Implemented Log Transformations to handle skewed target variables and One-Hot Encoding for 43 categorical features.
2. **Persistence:** Exported the trained model and feature maps using `joblib` for real-time inference.
3. **Production UI:** Developed a Streamlit dashboard allowing users to input house specs and receive instant valuations.

## 📂 Project Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: Exploratory Data Analysis and model experiments.
- `src/`: Production scripts and saved model files.
- `venv/`: Virtual environment.

## 🏃 How to Run
1. Clone the repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run src/app.py`