# Phising link detector - CN6000 Project

## Important run note
Do not open `ui/templates/index.html` as the main app if you want the real ML prediction. Run the Flask backend instead:

```bash
pip install -r requirements.txt
python src/train_models.py
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The HTML page has a safe fallback preview, but the main evidence comes from the Python backend and trained model.

---

# CN6000 Project: Detection and Analysis of Phishing Attacks Using Machine Learning

Student Number: 2673346  
Programme: Cybersecurity and Computer Networking  
Supervisor: Mahmud Ahmed  

## Project Overview

This project investigates phishing URL detection using machine learning. It extracts URL-based features, trains machine learning models, compares them with a heuristic baseline, and provides a clean web interface connected to the trained Python model.

The project is designed for defensive and educational use only. It does not send phishing emails, collect real credentials, or interact with real websites.

## Main Components

- `src/feature_extraction.py` — extracts URL features
- `src/train_models.py` — trains and saves the best ML model
- `app.py` — Flask backend API connected to the ML model
- `ui/templates/index.html` — HTML interface
- `ui/static/css/styles.css` — responsive UI styling
- `ui/static/js/app.js` — frontend logic
- `data/sample_urls.csv` — sample labelled dataset
- `models/phishing_model.joblib` — saved model after training
- `outputs/` — evaluation results after training

## How to Run the Full System

Open a terminal inside the project folder.

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

```bash
python src/train_models.py
```

This creates:

```text
models/phishing_model.joblib
models/model_metadata.json
outputs/model_results.csv
outputs/classification_reports.txt
outputs/f1_score_comparison.png
```

### 3. Start the Web Application

```bash
python app.py
```

### 4. Open the UI

Go to:

```text
http://127.0.0.1:5000
```

## How the UI Connects to the ML Model

The HTML form sends the URL to the Flask backend endpoint:

```text
/api/predict
```

The backend then:

1. extracts URL features using `feature_extraction.py`
2. loads the saved scikit-learn model from `models/phishing_model.joblib`
3. predicts whether the URL is phishing or legitimate
4. returns the prediction, confidence score, and extracted features to the HTML UI

## Dataset Format

The dataset should use this CSV format:

```csv
url,label
https://example.com,0
http://verify-account-example-login.test/security-update,1
```

Where:

- `0` = legitimate
- `1` = phishing

## Ethical Safety

This project must not:

- collect real passwords
- trick users without approval
- impersonate organisations for real-world targeting
- send phishing emails to real people without formal approval
- store sensitive personal data

The prototype performs defensive URL analysis only.
