from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI
import re
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

from keras.models import load_model
import pickle
import os


from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi.responses import FileResponse
import numpy as np 



"""
1. We are going to make some constants like: 
A. Model Path (BIGR)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels 
E. Emotion emojis 
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "BiGRU_Model.keras")
Tokenizer_path = os.path.join(BASE_DIR, "..", "tokenizer.pkl")
max_seq_len = 50

emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']    
emotion_emojis = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😄',
    'love': '😍',
    'sadness': '😢',
    'surprise': '😲'
}

"""
Preprocess the Upcoming Tex
"""

def preprocess_text(text:str)->list[str]:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [text]


"""
Request and Response Schemas
"""

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="The Sentence to analyze",
                      json_schema_extra={
                          "example": "I am so happy today!"}
                     )

class PredictionResponse(BaseModel):
    text: str 
    predicted_emotion: str
    confidence: float
    all_probabilites: dict[str, float]

class HealthResponse(BaseModel):
    status:str 
    model_loaded: bool


"""
4. Model loading and LifeSpan Management 
"""

dl_model ={}

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Loading Model and Tokenizer...")

    dl_model["BiGRU"] = load_model(model_path)
    with open(Tokenizer_path, "rb") as file:
        dl_model["tokenizer"]  = pickle.load(file)

    print('Model are loadded successfully...')

    yield

    dl_model.clear()


app = FastAPI(lifespan=lifespan)


   
"""
Mount the Static Files to the FASAPI APP 
A. Enable Cors (Cross-Origin Resource Sharing ) 
"""


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.mount('/static', StaticFiles(directory='static'), name='static')




"""
API ENDPOINTS
A. Server UI at Home page 

B. Health Check Endpoint 

C Predict Emotion Endpoints 

 """







@app.get("/", include_in_schema=False)


def server_ui():
    return FileResponse("static/index.html")



#B. Health Check Endpoint 

@app.get("/health", response_model=HealthResponse)


def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))




#C. Predict Emotion Endpoints

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    #1 Cleans the input sentences 

    BIGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("tokenizer")

    if BIGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.") 

    
    #2. Convert the words into numeric using the tokenizer 
    tokenized_text = tokenizer_model.texts_to_sequences([text_input.text])
    print("Tokenized Text:", tokenized_text)

    padded_sequence = pad_sequences(tokenized_text, maxlen=max_seq_len, padding='post', truncating='post')

    probalities = BIGRU_model.predict(padded_sequence)[0]

    top_emotion_index = int(np.argmax(probalities))

    all_probabilities ={
        label: float(prob) for label, prob in zip(emotion_labels, probalities)

    }



    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probalities[top_emotion_index]),
        all_probabilites = all_probabilities
    )
       

    



 