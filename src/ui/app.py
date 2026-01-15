"""
FAST Clinical Trial Screening UI

Optimized version with:
- Real-time progress updates
- 2-step workflow (vs 6 steps)
- ~60% faster screening
- Trial history with database persistence
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
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
# DATABASE - SUPABASE FOR TRIAL HISTORY PERSISTENCE
# =============================================================================

def get_supabase_client():
    """Get Supabase client for database operations."""
    try:
        from supabase import create_client, Client

        # Get credentials from secrets or env
        url = None
        key = None

        try:
            url = st.secrets.get("supabase", {}).get("url")
            key = st.secrets.get("supabase", {}).get("key")
        except Exception:
            pass

        if not url:
            url = os.getenv("SUPABASE_URL")
        if not key:
            key = os.getenv("SUPABASE_KEY")

        if url and key:
            return create_client(url, key)
    except Exception as e:
        print(f"Supabase not available: {e}")

    return None


def save_trial_to_db(trial_id: str, protocol: str = None, result: dict = None):
    """Save trial to database for history."""
    client = get_supabase_client()

    if client:
        try:
            data = {
                "trial_id": trial_id,
                "protocol_text": protocol[:5000] if protocol else None,  # Limit size
                "screening_result": json.dumps(result) if result else None,
                "created_at": datetime.now().isoformat()
            }
            client.table("trial_history").upsert(data, on_conflict="trial_id").execute()
            return True
        except Exception as e:
            print(f"Error saving to DB: {e}")

    # Fallback: save to session state
    if "trial_history_local" not in st.session_state:
        st.session_state.trial_history_local = []

    # Add to local history (avoid duplicates)
    existing_ids = [t["trial_id"] for t in st.session_state.trial_history_local]
    if trial_id not in existing_ids:
        st.session_state.trial_history_local.append({
            "trial_id": trial_id,
            "created_at": datetime.now().isoformat(),
            "has_result": result is not None
        })
        # Keep only last 50
        st.session_state.trial_history_local = st.session_state.trial_history_local[-50:]

    return False


def get_trial_history() -> List[dict]:
    """Get trial history from database or session."""
    client = get_supabase_client()

    if client:
        try:
            response = client.table("trial_history").select("trial_id, created_at").order("created_at", desc=True).limit(50).execute()
            return response.data
        except Exception as e:
            print(f"Error fetching history: {e}")

    # Fallback: return local history
    return st.session_state.get("trial_history_local", [])


def get_trial_from_db(trial_id: str) -> Optional[dict]:
    """Get specific trial data from database."""
    client = get_supabase_client()

    if client:
        try:
            response = client.table("trial_history").select("*").eq("trial_id", trial_id).single().execute()
            return response.data
        except Exception:
            pass

    return None


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
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "patient_validated" not in st.session_state:
    st.session_state.patient_validated = False
if "trial_history_local" not in st.session_state:
    st.session_state.trial_history_local = []
if "selected_trial_id" not in st.session_state:
    st.session_state.selected_trial_id = ""


def clear_session():
    """Clear all session state for new patient."""
    st.session_state.screening_result = None
    st.session_state.patient_data = None
    st.session_state.patient_validated = False


def validate_patient_data(data: dict) -> tuple[bool, str]:
    """Validate patient data structure (backend validation, no display)."""
    required_fields = ["patient_id", "age", "sex"]

    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    if not isinstance(data.get("age"), (int, float)) or data["age"] < 0:
        return False, "Invalid age value"

    if data.get("sex") not in ["male", "female", "other"]:
        return False, "Invalid sex value"

    return True, "Valid"


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

# CLEAR/RESET BUTTON
st.sidebar.markdown("---")
if st.sidebar.button("Clear & New Patient", type="secondary", use_container_width=True):
    clear_session()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("Trial Selection")

# =============================================================================
# TRIAL HISTORY (LEFT SIDEBAR)
# =============================================================================

trial_history = get_trial_history()

if trial_history:
    st.sidebar.subheader("Recent Trials")

    # Display history as selectable buttons
    for i, trial in enumerate(trial_history[:10]):  # Show last 10
        trial_id_hist = trial.get("trial_id", "Unknown")
        created = trial.get("created_at", "")[:10]  # Just date

        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button(f"{trial_id_hist}", key=f"hist_{i}", use_container_width=True):
                st.session_state.selected_trial_id = trial_id_hist
                st.rerun()
        with col2:
            st.caption(created)

    st.sidebar.markdown("---")

# Trial input section
trial_source = st.sidebar.radio(
    "Protocol Source",
    ["Enter Trial ID", "Paste Protocol"]
)

trial_id = ""
trial_protocol = ""

# Use selected trial from history if available
default_trial = st.session_state.get("selected_trial_id", "")

if trial_source == "Enter Trial ID":
    trial_id = st.sidebar.text_input("Trial ID", value=default_trial, placeholder="NCT12345678")
    st.sidebar.info("Will use default protocol template")
else:
    trial_id = st.sidebar.text_input("Trial ID", value=default_trial, placeholder="NCT12345678")
    trial_protocol = st.sidebar.text_area("Protocol Text", height=200)

# Clear selected after use
if trial_id and trial_id != default_trial:
    st.session_state.selected_trial_id = ""


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
tab1, tab2, tab3 = st.tabs(["Patient Form", "JSON Input", "Batch Processing"])

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

        # Backend validation (no JSON display to user)
        is_valid, msg = validate_patient_data(patient_data)
        if is_valid:
            st.session_state.patient_data = patient_data
            st.session_state.patient_validated = True
            st.success(f"Patient {patient_id} loaded successfully!")
        else:
            st.error(f"Validation error: {msg}")


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

    if st.button("Load Patient", key="parse"):
        try:
            parsed_data = json.loads(json_input)
            # Backend validation (no JSON display)
            is_valid, msg = validate_patient_data(parsed_data)
            if is_valid:
                st.session_state.patient_data = parsed_data
                st.session_state.patient_validated = True
                st.success(f"Patient {parsed_data.get('patient_id', 'N/A')} loaded successfully!")
            else:
                st.error(f"Validation error: {msg}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")


with tab3:
    st.header("Batch Processing")
    st.markdown("Process multiple patients at once (up to 300+)")

    batch_json = st.text_area(
        "Patients JSON Array",
        value='[{"patient_id": "PT001", "age": 58, "sex": "male", "diagnoses": [], "medications": [], "lab_values": []}]',
        height=200,
        help="Paste a JSON array of patient objects"
    )

    batch_trial_id = st.text_input("Trial ID for Batch", value="NCT12345678", key="batch_trial")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Batch Screening", type="primary"):
            try:
                patients = json.loads(batch_json)
                if not isinstance(patients, list):
                    st.error("Input must be a JSON array")
                else:
                    st.session_state.batch_results = []
                    progress_bar = st.progress(0)
                    status = st.empty()

                    for i, patient in enumerate(patients):
                        # Validate each patient
                        is_valid, msg = validate_patient_data(patient)
                        if not is_valid:
                            st.session_state.batch_results.append({
                                "patient_id": patient.get("patient_id", f"Patient_{i}"),
                                "decision": "ERROR",
                                "error": msg
                            })
                            continue

                        status.info(f"Processing {patient.get('patient_id', f'Patient_{i}')} ({i+1}/{len(patients)})")
                        progress_bar.progress((i + 1) / len(patients))

                        # Run screening (simplified for batch)
                        try:
                            from src.agents.supervisor_fast import FastSupervisorAgent
                            agent = FastSupervisorAgent()

                            protocol = f"""CLINICAL TRIAL: {batch_trial_id}
                            INCLUSION: Age 18-75, Type 2 Diabetes, HbA1c 7-10%
                            EXCLUSION: Type 1 Diabetes, Pregnancy, Renal impairment"""

                            loop = asyncio.new_event_loop()
                            result = loop.run_until_complete(
                                agent.screen_patient(patient, protocol, batch_trial_id)
                            )
                            loop.close()

                            st.session_state.batch_results.append({
                                "patient_id": patient.get("patient_id"),
                                "decision": result.get("decision", "UNKNOWN"),
                                "confidence": result.get("confidence", 0),
                                "narrative": result.get("clinical_narrative", "")[:100]
                            })
                        except Exception as e:
                            st.session_state.batch_results.append({
                                "patient_id": patient.get("patient_id"),
                                "decision": "ERROR",
                                "error": str(e)
                            })

                    progress_bar.progress(100)
                    status.success(f"Batch complete! {len(patients)} patients processed")

            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    with col2:
        if st.button("Clear Batch Results"):
            st.session_state.batch_results = []
            st.rerun()

    # Display batch results
    if st.session_state.batch_results:
        st.subheader("Batch Results")
        df = pd.DataFrame(st.session_state.batch_results)
        st.dataframe(df, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        decisions = [r.get("decision") for r in st.session_state.batch_results]
        with col1:
            st.metric("Eligible", decisions.count("ELIGIBLE"))
        with col2:
            st.metric("Ineligible", decisions.count("INELIGIBLE"))
        with col3:
            st.metric("Uncertain/Error", len(decisions) - decisions.count("ELIGIBLE") - decisions.count("INELIGIBLE"))

        # Export batch results
        if st.button("Export Batch CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"batch_screening_{batch_trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )


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

            # Save trial to history (database + local)
            save_trial_to_db(trial_id, trial_protocol, result)


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
