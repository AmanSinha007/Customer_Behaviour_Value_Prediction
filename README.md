# Customer Behaviour & Value Prediction

A machine learning project for analyzing customer purchasing behaviour and predicting two things:

- whether a customer is likely to churn
- the estimated lifetime value of a customer

The project uses transaction data from the Online Retail II dataset and provides a Streamlit dashboard for exploring the predictions.

## What the project does

The raw transaction data is first cleaned and converted into customer-level features. These features are then used to train two XGBoost models.

### Churn Prediction

The churn model predicts the probability of a customer becoming inactive based on their previous purchasing behaviour.

Current results:

- AUC: 0.7100
- Accuracy: 0.6598
- F1 Score: 0.6406

### Customer Lifetime Value

The second model predicts the expected value of a customer.

Current result:

- MAE: 584.61

The models are saved using Joblib and loaded by the Streamlit application.

## Dataset

The project uses the Online Retail II dataset.

The original dataset contains more than 500,000 transaction records. After cleaning the data, the working dataset contains around 407,000 records.

Some of the main columns in the dataset are:

- Invoice
- StockCode
- Description
- Quantity
- InvoiceDate
- Price
- Customer ID
- Country

Customer-level features are created from these transactions before training the models.

## Features

Some of the features used for modelling are based on:

- Recency
- Purchase frequency
- Total spending
- Average order value
- Quantity purchased
- Previous transaction behaviour

The exact features are generated in `build_models.py`.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Plotly
- Streamlit

## Project Structure

```text
Predictive-Customer-Intelligence/
│
|- app.py
|- build_models.py
|- requirements.txt
|- README.md
│
|- models/
│   |- trained model files
│
|- data/
    |- dataset files