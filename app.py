from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import re
import sys

app = Flask(__name__)
CORS(app)

# Global variables for model and vectorizer
model = None
vectorizer = None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_models():
    global model, vectorizer
    try:
        # Try loading with joblib first (better for scikit-learn)
        model = joblib.load('logistic_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        # test_phrase = "Apka 4G bundle khatam ho chuka hai.Aap ka aaj ka muft 50MB 3G Internet khatam ho chuka hai. Bundle hasil karnay k liyah 345 per call karen"
        # X_test = vectorizer.transform([clean_text(test_phrase)])
        # print(f"\nTest prediction for '{test_phrase}':")
        # print("Raw prediction:", model.predict(X_test)[0])
        # print("Probabilities:", model.predict_proba(X_test)[0])
        # print("=======================\n")    
        print("✅ Models loaded successfully with joblib!")
    except:
        try:
            # Fallback to pickle
            with open('logistic_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            # Debug info (previous code)
    #         print("\n=== MODEL DEBUG INFO ===")
    #         print("Model classes:", model.classes_)
    #         print("Vectorizer vocab size:", len(vectorizer.vocabulary_))
    
    # # NEW TEST PREDICTION CODE
    #         test_phrase = "win free money now"
    #         X_test = vectorizer.transform([clean_text(test_phrase)])
    #         print(f"\nTest prediction for '{test_phrase}':")
    #         print("Raw prediction:", model.predict(X_test)[0])
    #         print("Probabilities:", model.predict_proba(X_test)[0])
    #         print("=======================\n")    
                
            print("✅ Models loaded successfully with pickle!")
        except Exception as e:
            print(f"❌ Failed to load models: {str(e)}")
            sys.exit(1)

# Load models when starting the app
load_models()
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request"}), 400
        
        raw_message = data['message']
        cleaned_message = clean_text(raw_message)
        
        # Vectorize and predict
        X = vectorizer.transform([cleaned_message])
        prediction = model.predict(X)[0]  # This should be 0 or 1
        prediction_proba = model.predict_proba(X)
        
        # Debug output (view in terminal)
        print(f"\nInput: '{raw_message}'")
        print(f"Cleaned: '{cleaned_message}'")
        print(f"Raw prediction: {prediction}")
        print(f"Probabilities: {prediction_proba[0]}")
        
        # Determine label based on model's classes
        if hasattr(model, 'classes_'):
            # If model has classes info (most sklearn models)
            if model.classes_[0] == 0:  # Assuming [0,1] or ['ham','spam']
                label = "spam" if prediction == 1 else "ham"
            else:
                label = "spam" if prediction == model.classes_[1] else "ham"
        else:
            # Fallback for unknown model types
            label = "spam" if prediction == 1 else "ham"
        
        return jsonify({
            "prediction": label,
            "confidence": float(np.max(prediction_proba)),
            "original_message": raw_message
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)