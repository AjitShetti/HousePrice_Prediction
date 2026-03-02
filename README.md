# 🏠 House Price Prediction using Machine Learning

This project predicts house prices based on various features such as area, number of bedrooms, furnishing status, road access, and more.  
It follows a modular structure — with clean separation for data preprocessing, model training, and prediction.

---

## 🚀 Project Overview

The goal is to build a regression model that can estimate the price of a house using real-world housing data.  
The pipeline supports multiple algorithms including:
- Linear Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  

The model with the highest R² score is automatically saved for future predictions.

---

## 🧩 Project Structure

```text
main.py             # entry-point to train and evaluate models
app.py              # Streamlit web app for price prediction
src/                # python modules
  data_preprocessing.py
  model_training.py
  model_evaluation.py
```

## 🛠️ Running the Streamlit App

Once you have trained a model (e.g. by executing `python main.py`), start the interactive
UI with:

```bash
streamlit run app.py
```

This will open a browser window allowing users to input property details and receive a
predicted price.

## 🧩 Project Structure

