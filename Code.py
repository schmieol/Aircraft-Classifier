import os
from pathlib import Path

import streamlit as st
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

# -----------------------------
# Einstellungen
# -----------------------------

MODEL_NAME = "dima806/military_aircraft_image_detection"

OUTPUT_DIR = "sortierte_flugzeuge"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Modell laden
# -----------------------------

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    return processor, model


processor, model = load_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Flugzeug-Erkennung",
    layout="wide"
)

st.title("✈️ Flugzeugmodell-Erkennung")
st.write(
    "Lade mehrere Bilder hoch. "
    "Die App erkennt das Flugzeugmodell "
    "und sortiert die Bilder automatisch."
)

uploaded_files = st.file_uploader(
    "Bilder hochladen",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -----------------------------
# Bildklassifikation
# -----------------------------

def classify_image(image):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probs = torch.nn.functional.softmax(
        logits,
        dim=-1
    )

    predicted_idx = probs.argmax(-1).item()

    label = model.config.id2label[predicted_idx]

    confidence = probs[0][predicted_idx].item()

    return label, confidence

# -----------------------------
# Verarbeitung
# -----------------------------

if uploaded_files:

    st.subheader("Ergebnisse")

    cols = st.columns(3)

    for index, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")

        label, confidence = classify_image(image)

        # Ordner für Flugzeugtyp erstellen
        label_dir = os.path.join(
            OUTPUT_DIR,
            label
        )

        os.makedirs(label_dir, exist_ok=True)

        # Bild speichern
        save_path = os.path.join(
            label_dir,
            uploaded_file.name
        )

        image.save(save_path)

        # Anzeige
        with cols[index % 3]:

            st.image(
                image,
                caption=f"{label} ({confidence:.2%})",
                use_container_width=True
            )

            st.success(f"Erkannt: {label}")

            st.write(
                f"Confidence: {confidence:.2%}"
            )

            st.write(
                f"Gespeichert unter:\n{save_path}"
            )

# -----------------------------
# Ordnerübersicht
# -----------------------------

st.divider()

st.subheader("Sortierte Kategorien")

base_path = Path(OUTPUT_DIR)

if base_path.exists():

    for folder in sorted(base_path.iterdir()):

        if folder.is_dir():

            file_count = len(list(folder.glob("*")))

            st.write(
                f"📁 {folder.name} "
                f"({file_count} Bilder)"
            )
