# Airline-Dynamic-Pricing-Model


This repository contains a machine learning model for predicting and suggesting airline ticket prices based on various factors including temporal features, seat availability, competitor pricing, and demand forecasts.

**Overview**

The model uses Random Forest Regression to predict optimal ticket prices by considering multiple factors that influence airline pricing. It includes feature engineering for temporal data and handles categorical variables through one-hot encoding.

**Features Used**

Days before flight
Total seats available
Remaining tickets
Competitor pricing
Demand forecast
Day of week
Month
Weekend status
Holiday status
Season

**Requirements**

Copypandas
numpy
scikit-learn
Installation

**Clone this repository:**

bashCopygit clone https://github.com/AbhinavDubey30/Airline-Dynamic-Pricing-Model.git

cd airline-pricing-model

**Install required packages:**

bashCopypip install -r requirements.txt

**Usage**

Training the Model

The model is trained on historical airline pricing data. To train the model:

**Ensure your data is in CSV format with the following columns:**

date
departure_date
is_weekend
is_holiday
season
days_before_flight
total_seats
tickets_left
competitor_price
demand_forecast
our_price


**Run the training script:**

pythonCopypython train_model.py

**Making Predictions**

To get a price suggestion, use the suggest_dynamic_price function:
pythonCopypredicted_price = suggest_dynamic_price(
    days_before_flight=30,
    total_seats=200,
    tickets_left=150,
    competitor_price=250,
    demand_forecast=0.75,
    departure_date='2024-12-25',
    is_weekend=True,
    is_holiday=True,
    season='winter'
)

**Model Performance**
The model's performance is evaluated using:

Mean Squared Error (MSE)
R-squared Score

The actual performance metrics will be displayed after training.

**Data Preprocessing**
The model performs the following preprocessing steps:

Converts date columns to datetime format
Extracts day of week and month features
Creates dummy variables for categorical features
Scales numerical features using StandardScaler

Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
