import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
from datetime import datetime
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notebooks.predict_factory import MotorFailurePredictor
from notebooks.generate_report import generate_report, save_pdf

# Page Configuration
st.set_page_config(
    page_title="AI Industrial Monitoring Dashboard",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    config_path = ROOT / "models" / "production_config.json"
    if not config_path.exists():
        return None
    try:
        return MotorFailurePredictor.from_config(config_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = load_model()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏭 AI Industrial Monitoring Dashboard")
    st.markdown("### Maintenance Prédictive des Moteurs Asynchrones")
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Siemens-logo.svg/1200px-Siemens-logo.svg.png", width=150)

if predictor is None:
    st.error("❌ Model artifacts not found. Please run training first.")
    st.stop()

# Sidebar - Settings
st.sidebar.header("Paramètres")
manufacturer = st.sidebar.selectbox("Fabriquant SCADA", ["Siemens", "Schneider"])
motor_type = st.sidebar.selectbox("Type de Moteur", ["VIDA", "KIRICI", "POMPA", "ELEVATOR"])

# Tabs
tab1, tab2 = st.tabs(["📋 Surveillance en Temps Réel", "🔍 Analyse Détaillée"])

with tab1:
    st.subheader("Entrée des Données Capteurs")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        amb_temp = st.number_input("Temp. Ambiante (°C)", value=28.0, step=0.5)
        mot_temp = st.number_input("Temp. Moteur (°C)", value=38.5, step=0.5)
    with col_s2:
        rpm = st.number_input("Vitesse (RPM)", value=1500.0, step=10.0)
        torque = st.number_input("Couple (Nm)", value=40.0, step=1.0)
    with col_s3:
        hours = st.number_input("Heures de Marche (h)", value=1.5, step=0.1)
        m_id = st.text_input("ID Machine", value="MOTEUR_01")

    if st.button("Lancer le Diagnostic", type="primary"):
        # Predict
        raw_input = {
            "AmbientTemp": amb_temp,
            "MotorTemp": mot_temp,
            "Speed_RPM": rpm,
            "Torque_Nm": torque,
            "RunHours": hours
        }
        
        result = predictor.predict(raw_input, machine_id=m_id, motor_type=motor_type)
        
        st.divider()
        
        # Check if prediction was skipped (Anomaly or Validation Error)
        if result.get("skip_prediction"):
            st.warning(f"⚠️ {result['action']}")
            if "validation_warnings" in result:
                for w in result["validation_warnings"]:
                    st.error(f"Détail: {w}")
            if "anomaly_score" in result:
                st.info(f"Score d'anomalie: {result['anomaly_score']}")
            st.stop()

        # Results Display (Prediction succeeded)
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            # Reverting to the correct key name from predict_factory.py
            prob = result.get("failure_probability_pct", 0.0)
            risk = result["risk_level"]
            icon = result["risk_icon"]
            
            st.metric("Probabilité de Panne", f"{prob}%")
            
            color = "#2ecc71" if risk == "NORMAL" else "#f1c40f" if risk == "ATTENTION" else "#e67e22" if risk == "URGENT" else "#e74c3c"
            st.markdown(f"""
                <div class="status-card" style="background-color: {color}">
                    <h2 style="margin:0; text-align:center;">{icon} {risk}</h2>
                    <p style="margin:0; text-align:center;">{result['action']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"⏳ **Délai estimé :** {result['time_to_failure_estimate']}")

        with res_col2:
            st.write("#### Modes de Défaillance")
            modes = result["likely_failure_modes"]
            for m in modes:
                st.write(f"• {m}")
            
            st.write("#### Rapport Diagnostic")
            report = generate_report(result, manufacturer=manufacturer)
            
            # Generate PDF for download
            pdf_path = ROOT / "reports" / f"web_report_{m_id}.pdf"
            save_pdf(str(pdf_path), report)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le Rapport PDF",
                    data=f,
                    file_name=f"Rapport_{m_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            
            with st.expander("Voir le rapport texte"):
                st.text(report["text"])

with tab2:
    st.subheader("Tendances et Maintenance")
    st.write("Visualisation des données historiques et des tendances de dégradation.")
    
    # Simple table of recent reports
    st.write("#### Rapports Récents")
    summary_path = ROOT / "reports" / "summary_dashboard.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            if not df.empty:
                # Handle inconsistent naming between old/new files
                if "failure_probability_pct" in df.columns:
                    df = df.rename(columns={"failure_probability_pct": "failure_prob"})
                
                # Show columns that exist
                cols = [c for c in ["machine_id", "failure_prob", "risk_level", "time_to_failure_estimate"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True)
            else:
                st.info("Historique vide.")
    else:
        st.info("Aucun historique disponible. Lancez des diagnostics pour remplir ce tableau.")

st.sidebar.divider()
st.sidebar.info(f"Modèle: {predictor.model_name}")
st.sidebar.caption(f"Seuil de détection: {predictor.threshold}")
