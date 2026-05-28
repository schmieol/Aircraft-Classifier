import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# --------------------------------
# Seitenkonfiguration
# --------------------------------
st.set_page_config(
    page_title="Flugzeug-Erkennung",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flugzeug-Erkennungs-App")
st.write("Lade ein Bild hoch und die KI erkennt den Flugzeugtyp.")

# --------------------------------
# Gerät wählen (CPU/GPU)
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Klassen definieren
# Reihenfolge MUSS wie im Training sein
# --------------------------------
classes = [
    "Airbus A320",
    "Boeing 737",
    "Boeing 747",
    "Eurofighter Typhoon",
    "F-16",
    "Cessna 172"
]

# --------------------------------
# Modell laden
# --------------------------------
@st.cache_resource
def load_model():
    model = torch.load(
        "aircraft_classifier.pth",
        map_location=device
    )

    model.eval()
    model.to(device)

    return model

model = load_model()

# --------------------------------
# Bildtransformation
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------
# Datei-Upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Bild öffnen
    image = Image.open(uploaded_file).convert("RGB")

    # Bild anzeigen
    st.image(
        image,
        caption="Hochgeladenes Bild",
        use_container_width=True
    )

    # Bild vorbereiten
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Vorhersage
    with torch.no_grad():

        output = model(img_tensor)

        probabilities = F.softmax(output[0], dim=0)

        confidence, predicted = torch.max(
            probabilities,
            0
        )

    predicted_class = classes[predicted.item()]
    confidence_percent = confidence.item() * 100

    # Ergebnis anzeigen
    st.success(
        f"Erkanntes Flugzeug: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence_percent:.2f}%"
    )

    # Alle Wahrscheinlichkeiten
    st.subheader("Wahrscheinlichkeiten")

    for i, prob in enumerate(probabilities):
        st.write(
            f"{classes[i]}: {prob.item() * 100:.2f}%"
        )
        

