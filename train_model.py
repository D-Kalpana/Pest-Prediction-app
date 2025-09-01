import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# === Load dataset ===
# If your dataset is in Google Drive, mount first and update path
data = pd.read_csv("/content/drive/MyDrive/pest_attack_dataset.csv")

# === Split features/target ===
X = data.drop("Pest_Attack", axis=1)
y = data["Pest_Attack"]

# === Encode categorical columns ===
X = pd.get_dummies(X)

# === Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained with Accuracy: {acc:.2f}")

# === Save model + feature names together ===
joblib.dump((model, list(X.columns)), "pest_model.pkl")
print("✅ Model saved as pest_model.pkl")
