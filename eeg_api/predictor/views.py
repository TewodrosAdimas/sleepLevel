import numpy as np
import joblib
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Load model once (global)
model = joblib.load('predictor/model/svm_eeg_model.pkl')


@api_view(['POST'])
def predict_eeg(request):
    try:
        data = request.data.get("signal")

        # ✅ Validation
        if data is None:
            return Response({"error": "No signal provided"}, status=400)

        if len(data) != 4094:
            return Response({"error": "Signal must have 4094 values"}, status=400)

        signal = np.array(data).reshape(1, -1)

        # Prediction
        pred = model.predict(signal)[0]
        probs = model.predict_proba(signal)[0]
        classes = model.classes_

        # Sort descending
        sorted_probs = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        return Response({
            "predicted_class": int(pred),
            "probabilities": [
                {"class": int(cls), "prob": float(p)}
                for cls, p in sorted_probs
            ]
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)