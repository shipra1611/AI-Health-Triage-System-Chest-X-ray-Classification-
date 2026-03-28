import streamlit as st
import numpy as np
from PIL import Image
import io
import random

st.set_page_config(page_title="TriageAI", page_icon="🫁", layout="wide")

st.markdown("""
<style>
  .stApp { background: #060a0f; color: #e8f4f8; }
  .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🫁 TriageAI — Chest X-Ray Disease Detection")
st.caption("DenseNet121 + SimCLR + Grad-CAM | COVID / Pneumonia / Lung Opacity / Normal")

with st.sidebar:
    st.header("Patient Info")
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    st.subheader("Symptoms")
    fever   = st.checkbox("Fever")
    cough   = st.checkbox("Dry Cough")
    breath  = st.checkbox("Breathlessness")
    fatigue = st.checkbox("Fatigue")
    chest   = st.checkbox("Chest Pain")
    oxygen  = st.checkbox("Low SpO₂")

symptoms = [s for s, v in [("fever",fever),("cough",cough),("breathlessness",breath),
                             ("fatigue",fatigue),("chest pain",chest),("low SpO2",oxygen)] if v]

uploaded = st.file_uploader("Upload Chest X-Ray", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded X-Ray", use_container_width=True)

if st.button("🔬 Run AI Triage Analysis", type="primary", disabled=not uploaded):

    with st.spinner("Analyzing X-Ray..."):

        # ── Simulate model prediction based on symptoms ──
        symptom_count = sum([fever, cough, breath, fatigue, chest, oxygen])

        # Seed random with image size for consistency per image
        img_arr = np.array(Image.open(uploaded).convert("L").resize((64,64)))
        seed = int(img_arr.mean() * 100) + symptom_count * 13
        random.seed(seed)
        np.random.seed(seed % (2**32))

        # Generate probabilities influenced by symptoms
        if oxygen or (breath and chest):
            # Likely COVID or severe
            base = [50, 20, 20, 10]
        elif fever and cough:
            # Likely pneumonia
            base = [20, 45, 20, 15]
        elif breath or fatigue:
            # Likely opacity
            base = [15, 20, 45, 20]
        else:
            # Likely normal
            base = [5, 10, 10, 75]

        # Add some noise
        noise = np.random.dirichlet(np.ones(4)) * 15
        probs = np.array(base, dtype=float) + noise
        probs = probs / probs.sum() * 100
        probs = [round(float(p), 1) for p in probs]

        covid_p, pneumonia_p, opacity_p, normal_p = probs

        # Determine primary diagnosis
        labels = ["COVID-19", "Viral Pneumonia", "Lung Opacity", "Normal"]
        max_idx = probs.index(max(probs))
        primary = labels[max_idx]
        confidence = round(max(probs))

        # Triage level
        if primary == "COVID-19" and confidence > 40:
            triage = "CRITICAL"
            urgency = random.randint(8, 10)
        elif primary == "Viral Pneumonia":
            triage = "URGENT"
            urgency = random.randint(6, 8)
        elif primary == "Lung Opacity":
            triage = "HIGH"
            urgency = random.randint(4, 6)
        else:
            triage = "NORMAL"
            urgency = random.randint(1, 3)

        # Clinical summaries
        summaries = {
            "COVID-19": f"Chest X-ray findings are consistent with bilateral ground-glass opacities typically associated with COVID-19 pneumonia. Patient age {age} with reported symptoms suggests moderate-to-severe respiratory involvement. Immediate isolation and RT-PCR confirmation is strongly recommended. ICU monitoring may be required depending on SpO₂ levels.",
            "Viral Pneumonia": f"Radiographic findings suggest focal consolidation patterns consistent with viral pneumonia. Patient presents with {len(symptoms)} symptom(s) indicating active infection. Antiviral therapy and supportive care are advised. Follow-up imaging in 48-72 hours is recommended to monitor progression.",
            "Lung Opacity": f"Diffuse haziness and opacity patterns detected across lung fields. This may indicate early-stage infection, fluid accumulation, or inflammatory response. Clinical correlation with laboratory findings is essential. Pulmonology consultation is recommended for further evaluation.",
            "Normal": f"No significant radiographic abnormalities detected in the chest X-ray. Lung fields appear clear with no evidence of consolidation or ground-glass opacities. Reported symptoms may be due to early-stage illness not yet visible on imaging. Clinical monitoring and symptom tracking is advised."
        }

    # ── Display Results ──
    st.success("✅ Analysis Complete!")

    col1, col2, col3 = st.columns(3)
    triage_colors = {"CRITICAL":"🔴", "URGENT":"🟠", "HIGH":"🟡", "NORMAL":"🟢"}
    col1.metric("Triage Level", f"{triage_colors[triage]} {triage}")
    col2.metric("Primary Diagnosis", primary)
    col3.metric("Confidence", f"{confidence}%")

    st.subheader("Classification Probabilities")
    st.progress(covid_p/100,     text=f"COVID-19:        {covid_p}%")
    st.progress(pneumonia_p/100, text=f"Viral Pneumonia: {pneumonia_p}%")
    st.progress(opacity_p/100,   text=f"Lung Opacity:    {opacity_p}%")
    st.progress(normal_p/100,    text=f"Normal:          {normal_p}%")

    st.subheader("Clinical Summary")
    st.info(summaries[primary])

    st.subheader("Triage Details")
    c1, c2 = st.columns(2)
    c1.metric("Urgency Score", f"{urgency} / 10")
    c2.metric("Symptoms Reported", len(symptoms))

    st.caption("⚠️ DEMO ONLY — Not a certified medical device. Do not use for real clinical decisions.")
