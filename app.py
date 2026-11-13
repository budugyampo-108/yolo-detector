import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Image Detector")

MODELS = {
    "Model 1 (exp-5)": "exp5.pt",
    "Model 2 (exp-6)": "exp6.pt",
    "Model 3 (exp-8)": "exp8.pt",
}

@st.cache_resource
def load_models():
    return {name: YOLO(path) for name, path in MODELS.items()}

models = load_models()
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image")
    found = False
    for name, model in models.items():
        result = model.predict(img, conf=0.25)[0]
        st.subheader(name)
        if len(result.boxes) == 0:
            st.write("No match (0%)")
            continue
        found = True
        st.image(result.plot()[:, :, ::-1])
        for box in result.boxes:
            label = model.names[int(box.cls)]
            st.write(f"**{label}**: {float(box.conf) * 100:.1f}% confidence")
    if not found:
        st.warning("None of the trained models recognised this image.")