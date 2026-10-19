from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st
import tensorflow as tf

IMG_SIZE = (224, 224)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {
    "glioma": "Gliome",
    "meningioma": "Méningiome",
    "notumor": "Pas de tumeur",
    "pituitary": "Tumeur pituitaire (hypophyse)",
}
MODEL_CANDIDATES = [
    "brainscan_model.keras",
    "best_model.keras",
    "brainscan_model.h5",
]

st.set_page_config(page_title="BrainScan", page_icon="🧠", layout="centered")


@st.cache_resource
def load_modele():
    base = Path(__file__).parent.parent
    model_path = base / "models" / "best_model.keras"
    if model_path.exists():
        return tf.keras.models.load_model(model_path), "best_model.keras"
    return None, None


def preprocess(file_bytes: bytes):
    """Même prétraitement que Preprocessor : resize 224x224, BGR->RGB, /255."""
    img = cv.imdecode(np.frombuffer(file_bytes, np.uint8), cv.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Image illisible ou corrompue.")
    img = cv.resize(img, IMG_SIZE)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    x = np.expand_dims(img.astype("float32") / 255.0, axis=0)
    return img, x


def predict_image(model, x):
    probs = model.predict(x, verbose=0)[0]
    k = int(np.argmax(probs))
    return CLASS_NAMES[k], float(probs[k]), probs


st.title("🧠 BrainScan")
st.caption("Classification d'images IRM cérébrales par CNN : glioma, meningioma, pituitary, notumor")

model, model_name = load_modele()
if model is None:
    st.error(
        "Modèle introuvable. Mets `brainscan_model.keras` (ou `best_model.keras` / "
        "`brainscan_model.h5`) dans le même dossier que `app.py`."
    )
    st.stop()
st.sidebar.success(f"Modèle chargé : {model_name}")

uploaded = st.file_uploader("Choisis une image IRM", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    try:
        img_rgb, x = preprocess(uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Image (224×224)", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        label, conf, probs = predict_image(model, x)

    with col2:
        st.subheader(CLASS_LABELS[label])
        st.metric("Confiance", f"{conf * 100:.1f} %")
        if label == "notumor":
            st.success("Aucune tumeur détectée par le modèle.")
        else:
            st.warning("Tumeur suspectée par le modèle.")

    st.markdown("#### Probabilités par classe")
    st.bar_chart({CLASS_LABELS[c]: float(p) for c, p in zip(CLASS_NAMES, probs)})

st.divider()
st.caption(
    "⚠️ Projet pédagogique : ce modèle ne remplace pas un diagnostic médical. "
    "Seul un professionnel de santé peut interpréter une IRM."
)