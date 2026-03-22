import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_excel("data/data.xlsx", engine="openpyxl")
model = joblib.load("models/fraud_model.pkl")

X = df.drop("Category", axis=1)

# Get feature importance
importances = model.feature_importances_
features = X.columns

fi_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(fi_df)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(fi_df["Feature"], fi_df["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance - Fraud Detection Model")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()