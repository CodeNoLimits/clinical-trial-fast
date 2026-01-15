"""
FAST Clinical Trial Screening UI

Optimized version with:
- Real-time progress updates
- 2-step workflow (vs 6 steps)
- ~60% faster screening
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import time

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env only if exists (local dev)
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

def get_api_key() -> str:
    """Get Google API key - prioritize st.secrets for Streamlit Cloud."""
    # Source 1: Streamlit secrets (PRIORITY)
    try:
        api_key = st.secrets.get("google", {}).get("api_key")
        if api_key and api_key != "your_gemini_api_key_here":
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
    except Exception:
        pass

    # Source 2: Environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        return api_key

    # Source 3: Session state
    if "google_api_key" in st.session_state and st.session_state.google_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
        return st.session_state.google_api_key

    return None


def init_api_key_sidebar():
    """Initialize API key input in sidebar if needed."""
    api_key = get_api_key()

    if not api_key:
        st.sidebar.error("API Key Required")
        user_key = st.sidebar.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google Gemini API key",
            key="api_key_input"
        )
        if user_key:
            st.session_state.google_api_key = user_key
            os.environ["GOOGLE_API_KEY"] = user_key
            st.sidebar.success("API Key configured!")
            st.rerun()
        return False

    st.sidebar.success("API Key configured")
    return True


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Clinical Trial Screening (FAST)",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .eligibility-eligible {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white; padding: 20px; border-radius: 10px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .eligibility-ineligible {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        color: white; padding: 20px; border-radius: 10px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .eligibility-uncertain {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: black; padding: 20px; border-radius: 10px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .fast-badge {
        background: linear-gradient(135deg, #6f42c1 0%, #007bff 100%);
        color: white; padding: 5px 15px; border-radius: 20px;
        font-size: 14px; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "screening_result" not in st.session_state:
    st.session_state.screening_result = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_eligibility_class(decision: str) -> str:
    return f"eligibility-{decision.lower()}"


def create_confidence_gauge(confidence: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6f42c1"},
            'steps': [
                {'range': [0, 70], 'color': "#dc3545"},
                {'range': [70, 80], 'color': "#ffc107"},
                {'range': [80, 90], 'color': "#17a2b8"},
                {'range': [90, 100], 'color': "#28a745"}
            ],
        }
    ))
    fig.update_layout(height=280)
    return fig


def run_fast_screening(patient_data: dict, trial_protocol: str, trial_id: str, progress_container) -> dict:
    """Run FAST screening with real-time progress updates."""
    try:
        from src.agents.supervisor_fast import FastSupervisorAgent
        agent = FastSupervisorAgent()

        progress_bar = progress_container.progress(0, text="Initializing...")
        status_text = progress_container.empty()

        current_progress = [0]  # Use list for mutable closure

        def update_progress(message: str):
            if "Step 1" in message:
                current_progress[0] = 30
            elif "Step 2" in message:
                current_progress[0] = 70
            progress_bar.progress(current_progress[0], text=message)
            status_text.info(f" {message}")

        async def _async_screen():
            return await agent.screen_patient(
                patient_data=patient_data,
                trial_protocol=trial_protocol,
                trial_id=trial_id,
                progress_callback=update_progress
            )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_async_screen())
        finally:
            loop.close()

        progress_bar.progress(100, text="Complete!")
        status_text.success(" Screening complete!")

        return result

    except Exception as e:
        st.error(f"Screening error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


# =============================================================================
# SIDEBAR
# =============================================================================

api_key_valid = init_api_key_sidebar()

st.sidebar.markdown("---")
st.sidebar.markdown('<span class="fast-badge">FAST MODE</span>', unsafe_allow_html=True)
st.sidebar.markdown("**2-step optimized workflow**")
st.sidebar.markdown("~60% faster than standard")

st.sidebar.markdown("---")
st.sidebar.title("Trial Selection")

trial_source = st.sidebar.radio(
    "Protocol Source",
    ["Enter Trial ID", "Paste Protocol"]
)

trial_id = ""
trial_protocol = ""

if trial_source == "Enter Trial ID":
    trial_id = st.sidebar.text_input("Trial ID", placeholder="NCT12345678")
    st.sidebar.info("Will use default protocol template")
else:
    trial_id = st.sidebar.text_input("Trial ID", placeholder="NCT12345678")
    trial_protocol = st.sidebar.text_area("Protocol Text", height=200)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title(" Clinical Trial Screening")
st.markdown('<span class="fast-badge">FAST MODE - 2-Step Workflow</span>', unsafe_allow_html=True)

st.markdown("""
**Optimized AI System** for rapid patient-trial matching.

| Original | Fast Mode |
|----------|-----------|
| 6 steps | 2 steps |
| ~25-35 sec | ~8-12 sec |
| 6 LLM calls | 2 LLM calls |
""")

# Patient input tabs
tab1, tab2 = st.tabs(["Patient Form", "JSON Input"])

with tab1:
    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        patient_id = st.text_input("Patient ID", value="PT001")
        age = st.number_input("Age", min_value=0, max_value=120, value=55)
        sex = st.selectbox("Sex", ["male", "female", "other"])

        st.subheader("Diagnoses")
        diagnosis = st.text_input("Primary Condition", placeholder="Type 2 Diabetes Mellitus")
        icd10 = st.text_input("ICD-10 Code", placeholder="E11.9")

    with col2:
        st.subheader("Medications")
        medication = st.text_input("Medication", placeholder="Metformin")
        dose = st.text_input("Dose", placeholder="1000mg twice daily")

        st.subheader("Lab Values")
        lab_test = st.text_input("Test Name", placeholder="HbA1c")
        lab_value = st.number_input("Value", value=8.0, format="%.1f")
        lab_unit = st.text_input("Unit", placeholder="%")

    if st.button("Build Patient Profile", key="build"):
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "diagnoses": [{"condition": diagnosis, "icd10": icd10}] if diagnosis else [],
            "medications": [{"drug_name": medication, "dose": dose}] if medication else [],
            "lab_values": [{"test": lab_test, "value": lab_value, "unit": lab_unit}] if lab_test else []
        }
        st.session_state.patient_data = patient_data
        st.success("Profile built!")
        st.json(patient_data)


with tab2:
    json_template = """{
    "patient_id": "PT001",
    "age": 58,
    "sex": "male",
    "diagnoses": [{"condition": "Type 2 Diabetes Mellitus", "icd10": "E11.9"}],
    "medications": [{"drug_name": "Metformin", "dose": "1000mg twice daily"}],
    "lab_values": [{"test": "HbA1c", "value": 8.2, "unit": "%"}]
}"""

    json_input = st.text_area("Patient JSON", value=json_template, height=300)

    if st.button("Parse JSON", key="parse"):
        try:
            st.session_state.patient_data = json.loads(json_input)
            st.success("JSON parsed!")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")


# =============================================================================
# SCREENING EXECUTION
# =============================================================================

st.divider()
st.header(" Run Fast Screening")

col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.patient_data:
        st.success("Patient data loaded")
    else:
        st.warning("Enter patient data above")

with col2:
    if trial_id:
        st.success(f"Trial: {trial_id}")
    else:
        st.warning("Enter trial ID")

# Progress container
progress_container = st.container()

if st.button(" Run Fast Screening", type="primary", use_container_width=True):
    if not api_key_valid:
        st.error("API Key required")
    elif not st.session_state.patient_data:
        st.error("Enter patient data first")
    elif not trial_id:
        st.error("Enter a trial ID")
    else:
        start_time = time.time()

        if not trial_protocol:
            trial_protocol = f"""
            CLINICAL TRIAL: {trial_id}

            INCLUSION CRITERIA:
            1. Age 18-75 years
            2. Diagnosis of Type 2 Diabetes Mellitus
            3. HbA1c between 7.0% and 10.0%
            4. Currently on stable metformin therapy

            EXCLUSION CRITERIA:
            1. Type 1 Diabetes
            2. Pregnant or nursing women
            3. Severe renal impairment (eGFR < 30 mL/min)
            """

        result = run_fast_screening(
            st.session_state.patient_data,
            trial_protocol,
            trial_id,
            progress_container
        )

        elapsed = time.time() - start_time

        if result:
            result["elapsed_time"] = f"{elapsed:.1f}s"
            st.session_state.screening_result = result


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

if st.session_state.screening_result:
    result = st.session_state.screening_result
    st.divider()
    st.header("Screening Results")

    # Timing badge
    elapsed = result.get("elapsed_time", "N/A")
    st.markdown(f'<span class="fast-badge">Completed in {elapsed}</span>', unsafe_allow_html=True)

    # Decision banner
    decision = result.get("decision", "UNCERTAIN")
    confidence = result.get("confidence", 0.0)

    st.markdown(
        f'<div class="{get_eligibility_class(decision)}">{decision}</div>',
        unsafe_allow_html=True
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confidence", f"{confidence:.0%}")
    with col2:
        st.metric("Level", result.get("confidence_level", "N/A"))
    with col3:
        review = "Yes" if result.get("requires_human_review") else "No"
        st.metric("Human Review", review)
    with col4:
        st.metric("Trial", trial_id)

    # Detailed tabs
    tab1, tab2, tab3 = st.tabs(["Analysis", "Explainability", "Narrative"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_confidence_gauge(confidence)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Key Factors:**")
            for factor in result.get("key_factors", []):
                st.markdown(f"- {factor}")
            if result.get("concerns"):
                st.markdown("**Concerns:**")
                for concern in result.get("concerns", []):
                    st.markdown(f"- {concern}")

    with tab2:
        data = result.get("explainability_table", [])
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No detailed data available")

    with tab3:
        narrative = result.get("clinical_narrative", "")
        if narrative:
            st.markdown(narrative)
        else:
            st.info("No narrative generated")

        # Export
        st.divider()
        if st.button("Export JSON"):
            st.download_button(
                "Download",
                json.dumps(result, indent=2),
                f"fast_screening_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown("""
**Clinical Trial Screening (FAST)** | v2.0.0 | Optimized 2-Step Workflow

Authors: CodeNoLimits | 2026
""")
