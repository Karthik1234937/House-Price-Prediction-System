
# 🏠 House Price Prediction using Machine Learning

This repository contains a comprehensive machine learning project focused on predicting house prices in Bengaluru, India. The goal is to build a robust regression model that accurately estimates the price of a house based on various features such as location, size, square footage, number of bathrooms, and more.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Preprocessing Steps](#preprocessing-steps)
- [Modeling](#modeling)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

## 📖 Overview

Accurately predicting real estate prices is a crucial aspect of market analysis for buyers, sellers, and investors. This project uses machine learning regression techniques to predict house prices in Bengaluru by applying preprocessing, exploratory data analysis, feature engineering, and multiple regression models. 

---

## 🌟 Features

- Data cleaning and handling missing values
- Unit normalization for non-standard square footage entries (acres, yards, etc.)
- Feature engineering for 'price per sqft' and 'size_new'
- Outlier detection and removal
- Model comparison among 11 regression algorithms
- Evaluation using RMSE, R² score, and cross-validation
- Parameter tuning for best performance

---

## 🧾 Dataset

- **Name:** Bengaluru House Data
- **Source:** Kaggle / Public dataset
- **Format:** CSV
- **Rows:** ~13,000
- **Columns:** area_type, location, size, total_sqft, bath, balcony, availability, price, etc.

---

## 🛠 Technologies Used

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Evaluation:** mean_squared_error, cross_val_score, R² score
- **IDE:** Jupyter Notebook / Google Colab

---

## 🔍 Preprocessing Steps

- Dropped columns with excessive missing values (e.g., `society`)
- Imputed missing values using mean/mode
- Converted ranges and units in `total_sqft` to float values
- Extracted number of rooms from `size`
- Removed outliers based on logical rules:
  - Avg sqft per BHK should be >= 350
  - `bath` should not exceed `BHK + 2`
- Created new features:
  - `total_sqft_new`
  - `size_new`
  - `price_per_sqft`
- Encoded categorical variables using Label Encoding

---

## 🤖 Modeling

### Models Trained:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Huber Regressor
- ElasticNetCV
- Random Forest Regressor
- Extra Tree Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- Decision Tree Regressor
- XGBoost Regressor

### Metrics Used:
- Accuracy (R² Score)
- RMSE (Root Mean Squared Error)
- Cross-Validation Score

---

## 📊 Results

| Algorithm                   | Accuracy | RMSE  | CV Score |
|----------------------------|----------|-------|----------|
| Linear Regression          | 0.94     | 13.40 | 0.96     |
| Lasso Regression           | 0.94     | 13.39 | 0.96     |
| Ridge Regression           | 0.94     | 13.40 | 0.96     |
| Huber Regression           | 0.90     | 17.66 | 0.92     |
| ElasticNet CV              | 0.94     | 13.55 | 0.95     |
| Random Forest Regressor    | 0.99     | 2.51  | 0.97     |
| Extra Tree Regressor       | **1.00** | **2.68**  | **0.99**     |
| Gradient Boosting Regressor| 1.00     | 3.48  | 0.99     |
| Support Vector Regressor   | 0.95     | 13.25 | 0.75     |
| Decision Tree Regressor    | 0.99     | 4.29  | 0.98     |
| XGB Regressor              | 0.91     | 16.96 | 0.91     |

✅ **Best Model:** Extra Tree Regressor  
📌 Achieved highest accuracy with lowest RMSE and highest CV Score.

---

## 🚀 Future Improvements

- Deploy the best model using Flask or Django as a web application
- Include map-based features (e.g., distance from city center, schools, hospitals)
- Handle temporal features (year built, year sold)
- Use deep learning (ANNs) or stacking ensemble methods
- Connect with real estate APIs for live data integration

---

## ✅ Conclusion

This project showcases a full machine learning pipeline for predicting house prices with strong accuracy and data handling practices. From extensive data preprocessing and unit conversion to outlier filtering and feature engineering, every step contributes to the model's robustness. Through rigorous comparison across a variety of regression models, the Extra Tree Regressor stood out as the best-performing model. This project lays a strong foundation for deploying predictive analytics in the real estate sector and can be expanded for broader use cases like live pricing dashboards and decision support systems.
