import numpy as np
import os
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "BiGRU_Model.keras")
tokenizer_path = os.path.join(BASE_DIR, "..", "tokenizer.pkl")

model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = 50
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

test_inputs = [
    "I love you",
    "I love food", 
    "I am so happy today!",
    "I feel so alone and hopeless today.",
    "I am furious that they cancelled the trip at the last minute.",
    "I feel terrified when walking down dark alleyways alone.",
    "I was shocked and completely surprised by the unexpected gift!",
    "I cannot sleep, my heart keeps racing.",
    "I absolutely adore you, you mean the world to me all of this according to that make a plan"
]

for text in test_inputs:
    tokenized = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokenized, maxlen=max_seq_len, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_emotion = emotion_labels[top_idx]
    print(f'Input: {text}')
    print(f'Predicted: {top_emotion} (prob: {probs[top_idx]:.4f})')
    print(f'All probs: {dict(zip(emotion_labels, probs))}')
    print('---')