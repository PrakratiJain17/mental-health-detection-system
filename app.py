# ===========================
# Mental Health Detection App
# ===========================

print(" Starting Mental Health Detection App...")

# ---------------------------
# Imports
# ---------------------------
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr
from tensorflow import keras
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# ---------------------------
# Load Models
# ---------------------------
print(" Loading models...")

tokenizer = DistilBertTokenizer.from_pretrained("./text_model")
model_text = TFDistilBertForSequenceClassification.from_pretrained("./text_model")

audio_model = keras.models.load_model("audio_model.h5")

print(" Models loaded")

# ---------------------------
# Labels
# ---------------------------
text_labels = ["Not Depressed", "Depressed"]
audio_emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ---------------------------
# Audio Feature Extraction
# ---------------------------
def extract_features(audio_path, sr=22050, n_mfcc=40):
    try:
        audio, sr = librosa.load(audio_path, sr=sr, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        print("Audio feature error:", e)
        return None

# ===========================
# Prediction Functions
# ===========================
print(" Creating prediction functions...")

def predict_text_label(text):
    if not text or text.strip() == "":
        return None

    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )
    outputs = model_text(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    idx = np.argmax(probs)

    return text_labels[idx]


def predict_audio_label(audio_path):
    if audio_path is None:
        return None

    features = extract_features(audio_path)
    if features is None:
        return None

    features = features.reshape(1, 40, 1)
    preds = audio_model.predict(features, verbose=0)[0]
    emotion = audio_emotions[np.argmax(preds)]

    # Emotion → Depression mapping (NO scores shown)
    depressed_emotions = ["Sad", "Fear", "Disgust", "Neutral"]

    if emotion in depressed_emotions:
        return "Depressed"
    else:
        return "Not Depressed"

# ===========================
# Support Message
# ===========================
def get_support_message(label):
    if label == "Depressed":
        return (
            "🧠 **You may be experiencing emotional distress.**\n\n"
            "💙 You are not alone. Help is available.\n\n"
            "📞 **India Helplines:**\n"
            "- AASRA (24x7): +91-9820466726\n"
            "- KIRAN (Govt. of India): 1800-599-0019\n\n"
            "🌱 Consider talking to a trusted person or mental health professional."
        )
    else:
        return (
            "😊 **No strong signs of depression detected.**\n\n"
            "✨ Continue maintaining healthy mental well-being habits.\n"
            "🧘 Stay connected and seek support whenever needed."
        )

# ===========================
# Gradio Interface Functions
# ===========================
def analyze_text(text):
    label = predict_text_label(text)
    if label is None:
        return "⚠️ Please enter valid text", ""

    return label, get_support_message(label)


def analyze_audio(audio):
    label = predict_audio_label(audio)
    if label is None:
        return "⚠️ Please upload valid audio", ""

    return label, get_support_message(label)


def analyze_combined(text, audio):
    text_label = predict_text_label(text) if text else None
    audio_label = predict_audio_label(audio) if audio else None

    if text_label == "Depressed" or audio_label == "Depressed":
        final_label = "Depressed"
    else:
        final_label = "Not Depressed"

    return final_label, get_support_message(final_label)

# ===========================
# Gradio UI
# ===========================
print("🎨 Launching Gradio UI...")

with gr.Blocks(title="Mental Health Detection System") as demo:
    gr.Markdown("# 🧠 Mental Health Detection & Support System")
    gr.Markdown("**Multi-Modal Mental Health Analysis Using Text and Audio**")

    # -------- TEXT TAB --------
    with gr.Tab("📝 Text Analysis"):
        t_in = gr.Textbox(lines=4, label="Enter your thoughts")
        t_btn = gr.Button("Analyze Text")
        t_out = gr.Textbox(label="Result")
        t_sup = gr.Markdown()

        t_btn.click(analyze_text, t_in, [t_out, t_sup])

    # -------- AUDIO TAB --------
    with gr.Tab("🎤 Audio Analysis"):
        a_in = gr.Audio(type="filepath", label="Upload or Record Audio")
        a_btn = gr.Button("Analyze Audio")
        a_out = gr.Textbox(label="Result")
        a_sup = gr.Markdown()

        a_btn.click(analyze_audio, a_in, [a_out, a_sup])

    # -------- COMBINED TAB --------
    with gr.Tab("🔄 Combined Analysis"):
        c_text = gr.Textbox(lines=3, label="Enter Text")
        c_audio = gr.Audio(type="filepath", label="Upload or Record Audio")
        c_btn = gr.Button("Analyze Both")
        c_out = gr.Textbox(label="Final Result")
        c_sup = gr.Markdown()

        c_btn.click(analyze_combined, [c_text, c_audio], [c_out, c_sup])

demo.launch()
