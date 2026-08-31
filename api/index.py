from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Ensure NLTK data path and downloads exist inside serverless directories
nltk.download('stopwords')
nltk.download('punkt')

app = FastAPI(title="Spam Detection API")

def clean_input_text(text: str) -> str:
    """
    Lowercase, tokenize, stopword, and stem raw input message.
    """
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Loading pickle resources lazily
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError("Pickle binary resources could not be loaded. Ensure pipeline runs generated them.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

class MessageRequest(BaseModel):
    text: str

@app.post("/predict")
@app.post("/api/predict")
def predict(request: MessageRequest):
    try:
        cleaned = clean_input_text(request.text)
        features = vectorizer.transform([cleaned]).toarray()
        prediction = int(model.predict(features)[0])
        label = "Spam" if prediction == 1 else "Ham"
        
        return {
            "prediction": label,
            "label_code": prediction,
            "cleaned_text": cleaned,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def index():
    return {
        "status": "online",
        "description": "FastAPI Spam Classifier Serverless Endpoint for Vercel Deployment"
    }
