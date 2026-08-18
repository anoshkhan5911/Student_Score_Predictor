import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="🎓 AI Student Performance Predictor",
    layout="centered"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    /* Main container styling */
    .main-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .main-card h1 {
        color: #ffffff !important;
        margin-bottom: 8px;
        font-size: 30px;
    }
    .main-card p {
        color: #e0e8f9 !important;
        font-size: 15px;
        margin: 0;
    }
    
    /* Result Box Styling */
    .result-box {
        background: #ffffff;
        border: 2px solid #27ae60;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.15);
    }
    .result-title {
        color: #7f8c8d;
        font-size: 15px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-score {
        color: #27ae60;
        font-size: 42px;
        font-weight: 800;
        margin: 5px 0;
    }

    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #888888;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 15px;
        border-top: 1px solid #e5e5e5;
    }

    /* Circular Floating GitHub Button (Bottom Right) */
    .floating-github {
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 999999;
        background: #181717;
        color: #ffffff !important;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .floating-github:hover {
        transform: translateY(-4px) scale(1.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.45);
        background: #000000;
        color: #ffffff !important;
    }
    .floating-github svg {
        fill: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved models & assets
@st.cache_resource
def load_assets():
    model = joblib.load('student_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('model_features.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Header Banner
st.markdown("""
    <div class="main-card">
        <h1>🎓 Student Performance AI Predictor</h1>
        <p>Estimate final academic scores based on student habits and demographics</p>
    </div>
""", unsafe_allow_html=True)

# Input Section
st.subheader("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("⏱️ Study Hours / Week", min_value=0.0, max_value=168.0, value=15.0, step=0.5)
    attendance = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    age = st.number_input("🎂 Student Age", min_value=5, max_value=100, value=18, step=1)
    gender = st.selectbox("👤 Gender", ["Male", "Female", "Other"])

with col2:
    study_method = st.selectbox("📚 Primary Study Method", ["Notes", "Textbook", "Group study", "Coaching", "Mixed"])
    parent_edu = st.selectbox("🎓 Parent Education Level", ["High school", "Diploma", "Graduate", "Post graduate", "PhD", "No formal"])
    extra_act = st.selectbox("🏅 Extracurricular Activities", ["Yes", "No"])

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Button
if st.button("🚀 Generate Prediction", use_container_width=True, type="primary"):
    input_data = np.zeros(len(features))
    
    if 'study_hours' in features:
        input_data[features.index('study_hours')] = float(study_hours)
    if 'attendance_percentage' in features:
        input_data[features.index('attendance_percentage')] = float(attendance)
    if 'age' in features:
        input_data[features.index('age')] = int(age)
        
    def add_dummy_feature(column_prefix, value):
        col_name = f"{column_prefix}_{value}"
        if col_name in features:
            input_data[features.index(col_name)] = 1.0

    add_dummy_feature('gender', gender)
    add_dummy_feature('study_method', study_method)
    add_dummy_feature('parent_education', parent_edu)
    add_dummy_feature('extra_activities', extra_act)
    
    try:
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]
        prediction = max(0.0, min(float(prediction), 100.0))
        
        # Result Card Display
        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Predicted Final Score</div>
                <div class="result-score">{prediction:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar visual
        st.progress(prediction / 100.0)
        
        if prediction >= 80:
            st.balloons()
            st.success("🌟 Outstanding performance predicted! Keep it up!")
        elif prediction >= 50:
            st.info("👍 Good potential, steady practice will yield higher results!")
        else:
            st.warning("⚠️ Improvement needed in attendance and study hours.")
            
    except Exception as err:
        st.error(f"Prediction calculation failed: {err}")

# Footer Note
st.markdown("""
    <div class="footer-text">
        Built by <b>Anosh Dilshad</b> | Powered by Python & Streamlit
    </div>
""", unsafe_allow_html=True)

# Circular Floating Bottom-Right GitHub Button
st.markdown("""
    <a href="https://github.com/anoshkhan5911" target="_blank" class="floating-github" title="Visit Anosh's GitHub Profile">
        <svg height="26" width="26" viewBox="0 0 16 16">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
    </a>
""", unsafe_allow_html=True)