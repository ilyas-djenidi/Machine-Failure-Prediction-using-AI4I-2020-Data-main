import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from tests.synthetic_data_generator import SyntheticFactoryGenerator

# --- Configuration ---
st.set_page_config(page_title="Algerian Industrial Predictive Maintenance", layout="wide")

# --- Load the Trained Model ---
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'logistic_regression_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.getcwd(), 'models', 'logistic_regression_model.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
except Exception:
    model = None

# --- UI Setup ---
st.title("🇩🇿 Algerian Industrial Predictive Maintenance Dashboard")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🚀 Real-time Prediction", "📊 Model Insights", "🏭 Factory Simulator"])

# --- Tab 1: Real-time Prediction ---
with tab1:
    st.header("Predictive Diagnostic Tool")
    if model is None:
        st.error("Model not found. Please run tests/test_model_accuracy.py first.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sensor Input")
            c1, c2 = st.columns(2)
            air_temp = c1.number_input("Air Temp [K]", 290.0, 310.0, 300.0)
            proc_temp = c2.number_input("Process Temp [K]", 300.0, 320.0, 310.0)
            rot_speed = c1.number_input("Rotational Speed [rpm]", 1000.0, 2000.0, 1500.0)
            torque = c2.number_input("Torque [Nm]", 0.0, 100.0, 50.0)
            tool_wear = st.slider("Tool Wear [min]", 0, 250, 50)
            
        with col2:
            st.subheader("Action")
            if st.button("Analyze Machine State", use_container_width=True):
                # Ensure feature order matches training (logistic regression expects 5 features)
                # Note: original app used ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                input_df = pd.DataFrame([[air_temp, proc_temp, rot_speed, torque, tool_wear]], 
                                       columns=['Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]'])
                
                prediction = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
                
                if prediction == 1:
                    st.error("🚨 **CRITICAL: FAILURE RISK DETECTED**")
                    st.metric("Failure Probability", f"{prob[1]*100:.1f}%", delta="HIGH", delta_color="inverse")
                else:
                    st.success("✅ **STABLE: NORMAL OPERATION**")
                    st.metric("Failure Probability", f"{prob[1]*100:.1f}%", delta="LOW")

# --- Tab 2: Model Insights ---
with tab2:
    st.header("Performance Benchmarks")
    results_path = os.path.join(os.getcwd(), 'tests', 'results', 'model_performance.json')
    
    if not os.path.exists(results_path):
        st.warning("No performance data found. Run `python tests/test_model_accuracy.py` to populate.")
    else:
        with open(results_path, 'r') as f:
            data = json.load(f)
            
        st.write(f"**Best Model:** {data['best_model']}")
        
        # Display Metrics Table
        res_df = pd.DataFrame(data['results']).T
        st.dataframe(res_df.style.highlight_max(axis=0))
        
        # Plot Metrics
        fig = px.bar(res_df.reset_index(), x='index', y=['f1', 'accuracy', 'roc_auc'], 
                    barmode='group', title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrices
        st.subheader("Diagnostic Reliability (Confusion Matrices)")
        cols = st.columns(3)
        for i, m_name in enumerate(data['results'].keys()):
            img_path = f"tests/results/cm_{m_name.lower().replace(' ', '_')}.png"
            if os.path.exists(img_path):
                cols[i % 3].image(img_path, caption=m_name)

# --- Tab 3: Factory Simulator ---
with tab3:
    st.header("Factory Simulation & Stress Testing")
    st.info("Simulate multiple machines with synthetic failure signatures for pilot deployment testing.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_m = st.number_input("Number of Machines", 1, 50, 10)
        sim_days = st.slider("Simulation Period (Days)", 7, 180, 30)
        if st.button("Generate Simulation Data"):
            gen = SyntheticFactoryGenerator()
            with st.spinner("Generating factory telemetry..."):
                sim_df = gen.generate_factory(num_machines=num_m, days=sim_days)
                st.session_state['sim_data'] = sim_df
                
    if 'sim_data' in st.session_state:
        df = st.session_state['sim_data']
        st.success(f"Generated {len(df)} telemetry points for {num_m} machines.")
        
        # Overall health
        fail_count = df[df['failure_label'] == 1]['machine_id'].nunique()
        st.metric("Machines with Predicted Failures", f"{fail_count} / {num_m}")
        
        # Plot telemetry for a selected machine
        m_id = st.selectbox("Select Machine to Inspect", df['machine_id'].unique())
        m_df = df[df['machine_id'] == m_id]
        
        fig = px.line(m_df, x='timestamp', y=['temperature', 'vibration'], 
                     title=f"Telemetry Stream - {m_id}")
        
        # Mark failure points
        fails = m_df[m_df['failure_label'] == 1]
        for _, row in fails.iterrows():
            fig.add_annotation(x=row['timestamp'], y=row['temperature'], text="FAILURE", 
                             showarrow=True, arrowhead=1, bgcolor="red")
            
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button("Export Simulation results (CSV)", 
                          df.to_csv(index=False), 
                          "factory_simulation.csv", 
                          "text/csv")

st.markdown("---")
st.caption("Developed for Industrial Predictive Maintenance Systems 🇩🇿")