from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# 1) Load your artifacts
print("Loading model...")
model = joblib.load("model/xgb_model.pkl")
print("✅ Model loaded")

print("Loading scaler...")
scaler = joblib.load("model/scaler.pkl")
print("✅ Scaler loaded")

print("Loading features...")
feature_cols = joblib.load("model/feature_cols.pkl")
print("✅ Features loaded")
num_cols     = ['Age', 'CabinNum', 'TotalSpend', 'GroupSize']

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 API is up! Browse to /form"

@app.route("/form")
def form():
    return render_template("index.html", result=None)

@app.route("/submit", methods=["POST"])
def submit_form():
    # 2) Collect form inputs and build DataFrame
    data = {k: float(v) for k, v in request.form.items()}
    df   = pd.DataFrame([data])

    # 3) Scale numeric features
    df[num_cols] = scaler.transform(df[num_cols])

    # 4) Reindex to all features the model expects, filling missing cols with 0
    df = df.reindex(columns=feature_cols, fill_value=0)

    # 5) Predict
    pred = model.predict(df)[0]

    return render_template("index.html", result=bool(pred))

@app.route("/predict", methods=["POST"])
def predict_json():
    data = request.get_json()
    df   = pd.DataFrame([data])

    df[num_cols] = scaler.transform(df[num_cols])
    df = df.reindex(columns=feature_cols, fill_value=0)

    pred = model.predict(df)[0]
    return jsonify({"Transported": bool(pred)})


# # For local testing only
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)

