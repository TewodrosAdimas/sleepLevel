import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib

# Load model + scaler
model = joblib.load('predictor/model/svm_eeg_model.pkl')
scaler = joblib.load('predictor/model/scaler.pkl')


def home(request):
    return render(request, "predictor/index.html")


@csrf_exempt
def predict_api(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required"})

    try:
        data = json.loads(request.body)
        signal = np.array(data["signal"])
    except Exception:
        return JsonResponse({"success": False, "error": "Invalid JSON"})

    # Validate shape
    if signal.shape[0] != model.n_features_in_:
        return JsonResponse({
            "success": False,
            "error": f"Expected {model.n_features_in_} features, got {signal.shape[0]}"
        })

    # Reshape
    signal = signal.reshape(1, -1)

    # Apply scaler (IMPORTANT)
    signal = scaler.transform(signal)

    # Predict
    pred = model.predict(signal)[0]
    probs = model.predict_proba(signal)[0]
    classes = model.classes_

    sorted_probs = sorted(
        zip(classes, probs),
        key=lambda x: x[1],
        reverse=True
    )

    return JsonResponse({
        "success": True,
        "prediction": int(pred),
        "probabilities": [
            {"class": int(c), "prob": float(p)}
            for c, p in sorted_probs
        ]
    })