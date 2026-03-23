from django.shortcuts import render
import numpy as np
import joblib
from django.http import JsonResponse

model = joblib.load('predictor/model/svm_eeg_model.pkl')


def home(request):
    return render(request, "predictor/index.html")

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import numpy as np

@csrf_exempt
def predict_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        signal = np.array(data["signal"]).reshape(1, -1)

        pred = model.predict(signal)[0]
        probs = model.predict_proba(signal)[0]

        classes = model.classes_

        result = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        return JsonResponse({
            "prediction": int(pred),
            "probabilities": [
                {"class": int(c), "prob": float(p)} for c, p in result
            ]
        })

    return JsonResponse({"error": "POST request required"})