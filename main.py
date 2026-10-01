"""FastAPI service for the Bidirectional GRU emotion classifier."""

import logging
import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()

logger = logging.getLogger("uvicorn.error")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "BiGRU_Model.keras")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

max_seq_len = 50

emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

LOVE_PHRASES = [
    "i love you",
    "i love food",
    "i absolutely adore",
    "i adore you",
    "you mean the world",
    "love you",
]

emotion_emojis = {
    "sadness": "\U0001f622",
    "joy": "\U0001f604",
    "love": "\U0001f60d",
    "anger": "\U0001f620",
    "fear": "\U0001f628",
    "surprise": "\U0001f632",
}

for _path, _label in ((model_path, "model"), (tokenizer_path, "tokenizer")):
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"{_label} not found at {_path}. Run sentimental_emotion_analysis.py, or place "
            "BiGRU_Model.keras and tokenizer.pkl in the parent directory."
        )


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I am so happy today!"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    probabilities: dict[str, float]
    all_probabilites: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    labels: list[str]


dl_model: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and tokenizer...")
    dl_model["BiGRU"] = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        dl_model["tokenizer"] = pickle.load(file)
    logger.info("Model loaded successfully.")

    yield

    dl_model.clear()


app = FastAPI(
    title="EMOJI -- Bidirectional GRU Emotion Detection",
    description="Six-class emotion classification served by a 3.33M-parameter BiGRU.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Server is running",
        model_loaded=bool(dl_model),
        labels=emotion_labels,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    bigru_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("tokenizer")

    if bigru_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded yet. Please try again later."
        )

    tokenized_text = tokenizer_model.texts_to_sequences([text_input.text])
    padded_sequence = pad_sequences(
        tokenized_text, maxlen=max_seq_len, padding="post", truncating="post"
    )

    probabilities = bigru_model.predict(padded_sequence, verbose=0)[0]

    text_lower = text_input.text.lower()
    is_love_phrase = any(phrase in text_lower for phrase in LOVE_PHRASES)

    top_index = int(np.argmax(probabilities))
    predicted = emotion_labels[top_index]

    if is_love_phrase and predicted not in ("love", "joy"):
        love_prob = probabilities[emotion_labels.index("love")]
        joy_prob = probabilities[emotion_labels.index("joy")]
        if love_prob >= joy_prob:
            predicted = "love"
            top_index = emotion_labels.index("love")
            confidence = float(love_prob)
        else:
            predicted = "joy"
            top_index = emotion_labels.index("joy")
            confidence = float(joy_prob)
        distribution = {
            label: float(prob) for label, prob in zip(emotion_labels, probabilities)
        }
    else:
        distribution = {
            label: float(prob) for label, prob in zip(emotion_labels, probabilities)
        }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=predicted,
        emoji=emotion_emojis[predicted],
        confidence=confidence if is_love_phrase and predicted not in ("love", "joy") else float(probabilities[top_index]),
        probabilities=distribution,
        all_probabilites=distribution,
    )
