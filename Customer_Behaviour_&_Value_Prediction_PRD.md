# Predictive Customer Intelligence
## Product Requirements Document

## 1. Project Overview

This project uses historical e-commerce transaction data to understand customer behaviour and predict customer churn and customer lifetime value.

The main idea is to take transaction-level data and turn it into customer-level information that can be used for machine learning.

The project has two main prediction tasks:

1. Predict whether a customer is likely to churn.
2. Predict the expected lifetime value of a customer.

The results are shown through a Streamlit dashboard.

---

## 2. Problem Statement

E-commerce businesses have a large amount of transaction data, but transaction records alone do not give a clear picture of individual customers.

For example, it is difficult to directly identify:

- customers who may stop purchasing
- customers who generate high revenue
- customers who are becoming less active
- customers who may need more attention

This project tries to solve this by creating customer-level features from transaction history and using them for churn and CLV prediction.

---

## 3. Project Objectives

The main objectives are:

- Clean and prepare the retail transaction data.
- Convert transaction-level data into customer-level data.
- Create useful customer behaviour features.
- Build a churn prediction model.
- Build a CLV prediction model.
- Use SHAP to understand model predictions.
- Display the results through a Streamlit dashboard.

---

## 4. Dataset

The project uses the Online Retail II dataset.

The dataset contains transaction information including:

- Invoice number
- Stock code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

The original dataset contains more than 500,000 transaction records.

After cleaning and filtering the data, the working dataset contains approximately 407,000 records.

---

## 5. Data Processing

The raw transaction data is cleaned before it is used for modelling.

The processing includes:

- Handling missing customer IDs
- Removing invalid records
- Handling cancelled transactions
- Handling negative quantities
- Checking prices
- Converting invoice dates
- Removing duplicates where required
- Calculating transaction revenue

Revenue is calculated as:

```text
Revenue = Quantity × Unit Price