from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import pickle
import os
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Add the local directory to nltk data paths to prevent read-only filesystem download errors on Vercel
nltk_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nltk_data"))
nltk.data.path.insert(0, nltk_data_dir)

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
def serve_index():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_path = os.path.join(parent_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/style.css")
def serve_css():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    css_path = os.path.join(parent_dir, "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    return Response(content=css_content, media_type="text/css")

@app.get("/script.js")
def serve_js():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    js_path = os.path.join(parent_dir, "script.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    return Response(content=js_content, media_type="application/javascript")
