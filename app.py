import streamlit as st
import torch
from PIL import Image
import numpy as np
from scripts.predict import load_model
import torchvision.transforms as transforms

# --- App Config ---
st.set_page_config(
    page_title="Cassava Brown Streak Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa 0%, #f8bbd0 100%);
    }
    .stButton>button {
        color: white;
        background: #43a047;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1em;
        padding: 0.5em 2em;
        margin-top: 1em;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #43a047, #8bc34a);
    }
    .stAlert {
        border-radius: 8px;
    }
    .stFileUploader>div>div {
        border: 2px dashed #43a047;
        border-radius: 8px;
        background: #f1f8e9;
    }
    .stImage>img {
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(67,160,71,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown("""
<h1 style='text-align: center; color: #388e3c; font-size: 2.5em;'>🌿 Cassava Brown Streak Detection</h1>
<p style='text-align: center; color: #555; font-size: 1.2em;'>Upload a cassava leaf image to detect <b>brown streak</b>, <b>healthy</b>, or <b>mosaic disease</b> using a state-of-the-art deep learning model.</p>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def get_model():
    model, class_names = load_model('models/best_model.pth')
    return model, class_names

model, class_names = get_model()

# --- File Uploader ---
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a cassava leaf image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a cassava leaf for disease detection."
)

# --- Prediction Logic ---
def predict_image(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, 1)[0]
        pred = torch.argmax(probs).item()
        pred_class = class_names[pred]
        confidence = probs[pred].item()
    return pred_class, confidence, probs.cpu().numpy()

# --- Main App Logic ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner('Classifying...'):
        pred_class, confidence, all_probs = predict_image(image, model, class_names)
    st.success(f"Prediction: **{pred_class.replace('_', ' ').title()}** ({confidence*100:.2f}% confidence)")
    st.progress(int(confidence*100))
    # Show probability for all classes
    st.markdown("<h4 style='margin-top:2em;'>Class Probabilities</h4>", unsafe_allow_html=True)
    for i, cname in enumerate(class_names):
        st.write(f"{cname.replace('_', ' ').title()}: {all_probs[i]*100:.2f}%")
else:
    st.info("Please upload an image to get a prediction.")

# --- Footer ---
st.markdown("""
---
<div style='text-align:center; color:#888; font-size:1em;'>Cassava Brown Streak Detection &copy; 2025 | Powered by <b>PyTorch</b> & <b>Streamlit</b></div>
""", unsafe_allow_html=True)
