import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import numpy as np

# --------- Constants ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "model")

CLASSES = [
    'Healthy',
    'Cassava Mosaic Disease (CMD)',
    'Cassava Brown Streak Disease (CBSD)',
    'Cassava Green Mottle (CGM)',
    'Cassava Bacterial Blight (CBB)'
]

MEAN = [0.4326, 0.4952, 0.3120]
STD = [0.2179, 0.2214, 0.2091]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# --------- Model Definition ---------
def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, len(CLASSES))
    )
    return model

# --------- Load K-Fold Models ---------
@st.cache_resource
def load_all_models():
    models_list = []
    for fold in range(1, 6):
        model = get_model()
        path = os.path.join(MODELS_DIR, f"saved_model_fold{fold}.pth")
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])  # ✅ Fixed
        model.eval()
        models_list.append(model)
    return models_list

# --------- Ensemble Prediction ---------
def ensemble_predict(models, image):
    image = TRANSFORM(image).unsqueeze(0)
    preds = []
    with torch.no_grad():
        for model in models:
            output = model(image)
            probs = F.softmax(output, dim=1)
            preds.append(probs)
    avg_preds = torch.stack(preds).mean(dim=0)
    return avg_preds.squeeze().numpy()

# --------- Streamlit UI ---------
st.set_page_config(
    page_title="Cassava Leaf Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
    <style>
    body, .main { background-color: #f4f6f7; }
    .stButton > button {
        background-color: #2e8b57;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #246b44;
    }
    .stTitle, h1 {
        font-size: 2.5rem;
        color: #2e8b57;
        font-weight: 800;
        text-align: center;
    }
    .stImage > img {
        border-radius: 12px;
        margin-top: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Cassava Disease Detector")
st.markdown("Upload a **cassava leaf** image and let the AI predict its health status with confidence.")

uploaded_file = st.file_uploader("📤 Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Loading models and predicting..."):
        models = load_all_models()
        probs = ensemble_predict(models, image)

    predicted_class = CLASSES[np.argmax(probs)]
    confidence = np.max(probs) * 100

    st.markdown("## 🔍 Prediction Result")
    st.success(f"✅ **{predicted_class}**")
    st.info(f"🧪 Confidence: **{confidence:.2f}%**")

    st.markdown("## 📊 Class Probabilities")
    prob_df = pd.DataFrame({
        'Class': CLASSES,
        'Probability (%)': [round(p * 100, 2) for p in probs]
    }).sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)
    
    st.dataframe(prob_df, use_container_width=True)

    st.bar_chart(pd.DataFrame({'Probability': probs}, index=CLASSES))

else:
    st.warning("📎 Please upload an image to get a prediction.")
