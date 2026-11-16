import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
data = load_iris()

# Separate input and output
X = data.data
y = data.target

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model training completed.")
print("Trained model saved as model.pkl")