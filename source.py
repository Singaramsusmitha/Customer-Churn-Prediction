# ==================================================
# CUSTOMER CHURN PREDICTION
# ==================================================

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first few records
print("First 5 Rows:")
print(df.head())


# --------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------

# Remove customerID because it does not help in prediction
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges from object to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values with median
df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)


# --------------------------------------------------
# 3. Encode Categorical Features
# --------------------------------------------------

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])


# --------------------------------------------------
# 4. Define Features and Target
# --------------------------------------------------

X = df.drop("Churn", axis=1)   # Input features
y = df["Churn"]                # Target variable


# --------------------------------------------------
# 5. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 6. Train Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. Evaluate Model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 9. Feature Importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))


# --------------------------------------------------
# 10. Predict for a Sample Customer
# --------------------------------------------------

sample_customer = X.iloc[[0]]

prediction = model.predict(sample_customer)

result = (
    "Customer is likely to Churn"
    if prediction[0] == 1
    else "Customer is likely to Stay"
)

print("\nPrediction:")
print(result)
