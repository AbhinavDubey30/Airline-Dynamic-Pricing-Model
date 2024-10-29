import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv("airline_pricing_data.csv")

# Feature engineering
data["date"] = pd.to_datetime(data["date"])
data["departure_date"] = pd.to_datetime(data["departure_date"])
data["day_of_week"] = data["departure_date"].dt.dayofweek
data["month"] = data["departure_date"].dt.month

# Create dummy variables for categorical features
categorical_features = ["is_weekend", "is_holiday", "season"]
data_encoded = pd.get_dummies(data, columns=categorical_features)

# Select features for the model
features = [
    "days_before_flight",
    "total_seats",
    "tickets_left",
    "competitor_price",
    "demand_forecast",
    "day_of_week",
    "month",
] + [col for col in data_encoded.columns if col.startswith(tuple(categorical_features))]

X = data_encoded[features]
y = data_encoded["our_price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# Function to suggest dynamic price
def suggest_dynamic_price(
    days_before_flight,
    total_seats,
    tickets_left,
    competitor_price,
    demand_forecast,
    departure_date,
    is_weekend,
    is_holiday,
    season,
):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame(
        {
            "days_before_flight": [days_before_flight],
            "total_seats": [total_seats],
            "tickets_left": [tickets_left],
            "competitor_price": [competitor_price],
            "demand_forecast": [demand_forecast],
            "departure_date": [pd.to_datetime(departure_date)],
            "is_weekend": [is_weekend],
            "is_holiday": [is_holiday],
            "season": [season],
        }
    )

    # Feature engineering
    input_data["day_of_week"] = input_data["departure_date"].dt.dayofweek
    input_data["month"] = input_data["departure_date"].dt.month

    # Create dummy variables
    input_encoded = pd.get_dummies(
        input_data, columns=["is_weekend", "is_holiday", "season"]
    )

    # Ensure all columns from training are present
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Select and order features to match the training data
    input_features = input_encoded[X.columns]

    # Scale the features
    input_scaled = scaler.transform(input_features)

    # Make prediction
    predicted_price = rf_model.predict(input_scaled)[0]

    return predicted_price

