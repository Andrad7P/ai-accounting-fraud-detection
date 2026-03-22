import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_excel("data/data.xlsx", engine="openpyxl")

# Features and target
X = df.drop("Category", axis=1)
y = df["Category"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nModel Performance")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("\nModel saved to models/fraud_model.pkl")