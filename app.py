from flask import Flask, request, jsonify
import joblib

# Create Flask application
app = Flask(__name__)

# Load trained Machine Learning model
model = joblib.load("model.pkl")


# Home route
@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to Iris Flower Prediction API"
    })


# Health check route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Application is running"
    })


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "JSON object required"
        }), 400

    features = data.get("features")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": "Provide exactly four numeric features"
        }), 400

    try:
        features = [float(value) for value in features]
    except (ValueError, TypeError):
        return jsonify({
            "error": "All features must be numeric"
        }), 400

    prediction = model.predict([features])

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    result = classes[int(prediction[0])]

    return jsonify({
        "prediction": result
    })


# Run application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )