# app/app.py

import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\User\Desktop\Project_09\Models\hotel_cancellation_pipeline.joblib"

bundle = joblib.load(MODEL_PATH)

preprocessor = bundle["preprocessor"]
selected_features_idx = bundle["selected_features_idx"]
model = bundle["model"]


def predict_cancellation(input_data: dict):

    df = pd.DataFrame([input_data])

    X_processed = preprocessor.transform(df)
    X_selected = X_processed[:, selected_features_idx]

    pred = model.predict(X_selected)[0]
    proba = model.predict_proba(X_selected)[0][1]

    return {
        "is_canceled": int(pred),
        "cancel_probability": round(float(proba), 4)
    }

