import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import os

# --- Configuration ---
st.set_page_config(page_title="AI Health Triage System", page_icon="🩺", layout="wide")

DISEASES = ["COVID", "Viral Pneumonia", "Lung_Opacity", "Normal"]
MODEL_PATH = "best_model.pth"

# --- Define the Model Architecture ---
@st.cache_resource
def load_model():
    """Builds the DenseNet121 model and loads weights if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 4)
    )
    
    model_loaded = False
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model_loaded = True
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    
    model = model.to(device)
    model.eval()
    return model, device, model_loaded

model, device, is_model_loaded = load_model()

# --- Helpers ---
def preprocess_image(image_file):
    """Convert uploaded file into a 3-channel normalized tensor."""
    pil_image = Image.open(image_file)
    open_cv_image = np.array(pil_image.convert('L')) 
    image = cv2.resize(open_cv_image, (224, 224))
    image = np.stack([image, image, image], axis=-1)
    
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    
    image_tensor = torch.tensor(image).permute(2, 0, 1).float()
    return image_tensor

def predict(img_tensor, use_mock=False):
    """Run inference. If mock, return random probabilities."""
    if use_mock:
        probs = np.random.rand(4)
        return dict(zip(DISEASES, probs))
        
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    return dict(zip(DISEASES, probs))

def triage(pred_dict):
    """Assign triage urgency."""
    if pred_dict["COVID"] > 0.60:
        return "🔴 CRITICAL", f'COVID probability = {pred_dict["COVID"]:.1%}'
    if pred_dict["Viral Pneumonia"] > 0.55:
        return "🟠 URGENT", f'Viral Pneumonia probability = {pred_dict["Viral Pneumonia"]:.1%}'
    if pred_dict["Lung_Opacity"] > 0.50:
        return "🟡 HIGH", f'Lung Opacity probability = {pred_dict["Lung_Opacity"]:.1%}'
    return "🟢 NORMAL", "No critical findings detected."

# --- UI Layout ---
st.title("🩺 AI Health Triage System")
st.markdown("""
This application classifies Chest X-rays into **COVID-19**, **Viral Pneumonia**, **Lung Opacity**, or **Normal**.
Under the hood, it uses a DenseNet121 architecture pre-trained with self-supervised learning (**SimCLR**).
""")

if not is_model_loaded:
    st.warning("⚠️ **Model weights (`best_model.pth`) are missing locally.** The application is running in **MOCK MODE**, generating random probabilities for demonstration purposes.")

st.sidebar.header("Upload X-Ray")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Chest X-ray")
        st.image(uploaded_file, use_column_width=True)
        
    with col2:
        st.subheader("Triage Results")
        with st.spinner('Analyzing the image...'):
            img_tensor = preprocess_image(uploaded_file)
            predictions = predict(img_tensor, use_mock=not is_model_loaded)
            level, reason = triage(predictions)
            
            st.markdown(f"### Urgency Level: {level}")
            st.markdown(f"**Reason:** {reason}")
            st.divider()
            
            st.markdown("#### Disease Probabilities")
            for disease, prob in sorted(predictions.items(), key=lambda x: -x[1]):
                st.write(f"**{disease}**")
                st.progress(float(prob), text=f"{prob:.1%}")

else:
    st.info("👈 Upload a Chest X-ray image in the sidebar to begin analysis.")
