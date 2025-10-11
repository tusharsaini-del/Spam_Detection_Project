from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

SAMPLE_DATA = [
("Congratulations! You've won a free ticket. Click here.", 'spam'),
("Claim your prize now", 'spam'),
("Free vacation!!!", 'spam'),
("Hey, are we meeting today?", 'ham'),
("Can you send the assignment?", 'ham'),
("Let's have lunch tomorrow.", 'ham'),
]

def train_model(samples=SAMPLE_DATA):
    df = pd.DataFrame(samples, columns=['text','label'])
    X = df['text']
    y = df['label']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    pipeline.fit(X, y_enc)
    return pipeline, le

model, label_encoder = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    lang = request.form.get('lang', 'en')
    if not message.strip():
        return render_template('result.html', message=message, prediction='empty', prob=None, lang=lang)

    X = [message]
    probs = model.predict_proba(X)[0]
    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    # probability for predicted class
    pred_prob = float(probs[pred_idx])

    return render_template('result.html', message=message, prediction=pred_label, prob=pred_prob, lang=lang)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json or {}
    text = data.get('text', '')
    if not text.strip():
        return { 'error': 'empty_text' }, 400
    
    X = [text]
    probs = model.predict_proba(X)[0]
    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    pred_prob = float(probs[pred_idx])
    return { 'text': text, 'prediction': pred_label, 'probability': pred_prob }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
