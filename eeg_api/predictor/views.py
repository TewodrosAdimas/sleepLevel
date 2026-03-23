import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib

model = joblib.load('predictor/model/svm_eeg_model.pkl')


def home(request):
    return render(request, "predictor/index.html")


@csrf_exempt
def predict_api(request):
    if request.method == "POST":
        data = json.loads(request.body)

        signal = np.array(data["signal"]).reshape(1, -1)

        pred = model.predict(signal)[0]
        probs = model.predict_proba(signal)[0]
        classes = model.classes_

        sorted_probs = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        return JsonResponse({
            "prediction": int(pred),
            "probabilities": [
                {"class": int(c), "prob": float(p)}
                for c, p in sorted_probs
            ]
        })

    return JsonResponse({"error": "POST required"})