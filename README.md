
---

# 🧠 EEG AI Predictor (Django + Machine Learning)

A web-based **EEG signal classification system** that predicts brain states such as eyes open/closed and epileptic activity using a trained ML model (SVM + Scaler) served via a Django API.

---

## 🚀 Features

* 📡 Real-time EEG signal prediction
* 🤖 Machine Learning model (SVM)
* ⚖️ StandardScaler normalization (trained scaler.pkl)
* 🧠 Multi-class classification:

  * Eyes Open
  * Eyes Closed
  * Non-Epileptic Zone
  * Epileptic Zone
  * Seizure
* 📊 Probability confidence visualization
* 🎨 Modern dark UI dashboard
* ⚡ Fast REST API with Django

---

## 🧩 Tech Stack

### Backend

* Django
* Django REST (manual JSON API)
* NumPy
* Scikit-learn
* Joblib

### Frontend

* HTML5
* JavaScript (Vanilla)
* Bootstrap 5
* CSS3

---

## 🏗️ Project Structure

```
project/
│
├── predictor/
│   ├── model/
│   │   ├── svm_eeg_model.pkl
│   │   └── scaler.pkl
│   │
│   ├── views.py
│   ├── urls.py
│   └── templates/
│       └── index.html
│
├── manage.py
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the project

```bash
git clone https://github.com/your-username/eeg-ai-predictor.git
cd eeg-ai-predictor
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run migrations

```bash
python manage.py migrate
```

---

### 5. Start server

```bash
python manage.py runserver
```

---

## 🌐 Usage

1. Open browser:

```
http://127.0.0.1:8000/
```

2. Paste EEG signal or generate sample
3. Click **RUN PREDICTION**
4. View:

   * Predicted brain state
   * Confidence scores

---

## 📡 API Endpoint

### POST `/predict/`

### Request

```json
{
  "signal": [0.1, 0.2, 0.3, ...]
}
```

### Response

```json
{
  "prediction": 5,
  "probabilities": [
    {"class": 1, "prob": 0.05},
    {"class": 2, "prob": 0.10},
    {"class": 5, "prob": 0.80}
  ]
}
```

---

## 🧠 Class Mapping

| Class | Meaning            |
| ----- | ------------------ |
| 1     | Eyes Open          |
| 2     | Eyes Closed        |
| 3     | Non-Epileptic Zone |
| 4     | Epileptic Zone     |
| 5     | Seizure            |

---

## ⚠️ Notes

* Input must match **4094 EEG features**
* Model requires **same scaler used in training**
* Synthetic generated data is only for testing UI
* Real EEG data improves accuracy significantly

---

## 📈 Future Improvements

* Live EEG streaming support
* Seizure alert system (sound + popup)
* Database logging of predictions
* Patient dashboard
* Graph visualization (EEG waveform)
* Model retraining pipeline

---

## 👨‍💻 Author

Built by **Tewodros**

---

## 📜 License

This project is for educational and research purposes.

---


Just tell me 👍
