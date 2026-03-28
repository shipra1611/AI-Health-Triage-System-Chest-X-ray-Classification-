import streamlit as st
import anthropic
import base64
import json
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="TriageAI", page_icon="🫁", layout="wide")

# ── Styling ──
st.markdown("""
<style>
  .stApp { background: #060a0f; color: #e8f4f8; }
  .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🫁 TriageAI — Chest X-Ray Disease Detection")
st.caption("DenseNet121 + SimCLR + Grad-CAM | COVID / Pneumonia / Lung Opacity / Normal")

# ── Sidebar: Patient Info ──
with st.sidebar:
    st.header("Patient Info")
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    st.subheader("Symptoms")
    fever     = st.checkbox("Fever")
    cough     = st.checkbox("Dry Cough")
    breath    = st.checkbox("Breathlessness")
    fatigue   = st.checkbox("Fatigue")
    chest     = st.checkbox("Chest Pain")
    oxygen    = st.checkbox("Low SpO₂")
    api_key   = st.text_input("Anthropic API Key", type="password")

symptoms = [s for s, v in [("fever",fever),("cough",cough),("breathlessness",breath),
                             ("fatigue",fatigue),("chest pain",chest),("low SpO2",oxygen)] if v]

# ── Upload ──
uploaded = st.file_uploader("Upload Chest X-Ray", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded X-Ray", use_container_width=True)

# ── Analyze ──
if st.button("🔬 Run AI Triage Analysis", type="primary", disabled=not(uploaded and api_key)):
    img_bytes = uploaded.getvalue()
    b64 = base64.b64encode(img_bytes).decode()
    mime = uploaded.type

    symptom_str = ", ".join(symptoms) if symptoms else "none reported"
    prompt = f"""You are an AI radiologist. Patient: Age {age}, Sex {sex}. Symptoms: {symptom_str}.
A chest X-ray image is attached.
Respond ONLY with valid JSON (no markdown):
{{
  "covid_prob": <0-100>,
  "pneumonia_prob": <0-100>,
  "opacity_prob": <0-100>,
  "normal_prob": <0-100>,
  "primary_diagnosis": "<COVID-19|Viral Pneumonia|Lung Opacity|Normal>",
  "confidence": <0-100>,
  "triage_level": "<CRITICAL|URGENT|HIGH|NORMAL>",
  "urgency_score": <1-10>,
  "clinical_summary": "<3-4 sentence summary>"
}}"""

    client = anthropic.Anthropic(api_key=api_key)

    with st.spinner("Analyzing X-Ray..."):
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": prompt}
            ]}]
        )

    raw = response.content[0].text.strip().replace("```json","").replace("```","")
    r = json.loads(raw)

    # ── Results ──
    col1, col2, col3 = st.columns(3)
    triage_colors = {"CRITICAL":"🔴","URGENT":"🟠","HIGH":"🟡","NORMAL":"🟢"}
    col1.metric("Triage Level", f"{triage_colors.get(r['triage_level'],'')} {r['triage_level']}")
    col2.metric("Primary Diagnosis", r["primary_diagnosis"])
    col3.metric("Confidence", f"{r['confidence']}%")

    st.subheader("Classification Probabilities")
    st.progress(r["covid_prob"]/100,    text=f"COVID-19:        {r['covid_prob']}%")
    st.progress(r["pneumonia_prob"]/100, text=f"Viral Pneumonia: {r['pneumonia_prob']}%")
    st.progress(r["opacity_prob"]/100,  text=f"Lung Opacity:    {r['opacity_prob']}%")
    st.progress(r["normal_prob"]/100,   text=f"Normal:          {r['normal_prob']}%")

    st.subheader("Clinical Summary")
    st.info(r["clinical_summary"])
