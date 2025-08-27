# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import datetime as dt
import plotly.graph_objects as go

# ================================
# Load the trained pipeline (preprocessing + logistic regression)
# ================================
model_info = joblib.load("log_reg_model.pkl")
log_reg_model = model_info["model"]
threshold = model_info["threshold"]
top_features = model_info["features"]

# ================================
# Helper functions
# ================================
def calculate_age_r_lmp(dob, lmp_date):
    """Calculate age at last menstrual period (in years)."""
    return lmp_date.year - dob.year - ((lmp_date.month, lmp_date.day) < (dob.month, dob.day))

def calculate_p_stress(control, confid, yourway, overcome):
    """Perceived stress = sum of the 4 stress-related items."""
    return control + confid + yourway + overcome

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Depression Risk Classifier", layout="centered")
st.title("🧠 Depression Risk Classifier (Post-Menopausal Women)")
st.write(
    "Answer the questions below. The system will compute your risk of depression "
    "based on key lifestyle, health, and psychosocial factors."
)

# ---- Section 0: Demographics ----
st.subheader("Demographics")
dob = st.date_input(
    "Please enter your date of birth:",
    min_value=dt.date(1910, 1, 1),
    max_value=dt.date.today()
)
lmp_date = st.date_input(
    "What was the date of your last menstrual period?",
    min_value=dt.date(1910, 1, 1),
    max_value=dt.date.today()
)

# ---- Section 1: Symptoms ----
st.subheader("Symptoms (past two weeks)")
diffislp = st.radio("Difficulty sleeping?", ["No", "Yes"])
nisweat = st.radio("Night sweats?", ["No", "Yes"])
tense = st.radio("Feeling tense or nervous?", ["No", "Yes"])
irritab = st.radio("Irritability or grouchiness?", ["No", "Yes"])

# ---- Section 2: Perceived Stress Scale ----
st.subheader("Perceived Stress (past two weeks)")
scale_labels = {
    1: "Never", 2: "Almost Never", 3: "Sometimes", 4: "Fairly Often", 5: "Very Often"
}
control = st.selectbox("Felt unable to control important things in your life?", list(scale_labels.values()))
confid = st.selectbox("Felt confident about your ability to handle your personal problems?", list(scale_labels.values()))
yourway = st.selectbox("Felt things were going your way?", list(scale_labels.values()))
overcome = st.selectbox("Felt difficulties piling so high that you could not overcome them?", list(scale_labels.values()))

# ---- Section 3: Lifestyle ----
st.subheader("Lifestyle")
smoke_r = st.radio("Do you currently smoke cigarettes?", ["No", "Yes"])

# ---- Section 4: Health ----
st.subheader("Overall Health")
health_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
health = st.selectbox("Would you say your health in general is:", list(health_map.keys()))

# ================================
# Prediction
# ================================
if st.button("🔍 Predict Risk"):
    try:
        # Map categorical responses back to numeric
        diffislp_val = 2 if diffislp == "Yes" else 1
        nisweat_val = 2 if nisweat == "Yes" else 1
        tense_val = 2 if tense == "Yes" else 1
        irritab_val = 2 if irritab == "Yes" else 1
        smoke_val = 2 if smoke_r == "Yes" else 1
        health_val = health_map[health]

        # Reverse map Likert scale
        scale_reverse = {v: k for k, v in scale_labels.items()}
        control_val = scale_reverse[control]
        confid_val = scale_reverse[confid]
        yourway_val = scale_reverse[yourway]
        overcome_val = scale_reverse[overcome]

        # Derived features
        p_stress = calculate_p_stress(control_val, confid_val, yourway_val, overcome_val)
        age_r_lmp = calculate_age_r_lmp(dob, lmp_date)

        # Assemble input dataframe (must match training feature names!)
        input_data = pd.DataFrame([{
            "TENSE": tense_val,
            "IRRITAB": irritab_val,
            "CONTROL": control_val,
            "P_STRESS": p_stress,
            "YOURWAY": yourway_val,
            "NISWEAT": nisweat_val,
            "AGE_R_LMP": age_r_lmp,
            "SMOKE_R": smoke_val,
            "DIFFISLP": diffislp_val,
            "HEALTH": health_val
        }])

        # Use only top features
        input_data = input_data[top_features]

        # Prediction
        prob = log_reg_model.predict_proba(input_data)[:, 1][0]
        pred = int(prob >= threshold)

        # ---- Results ----
        st.subheader("Results")
        st.write(f"**Predicted Risk Probability:** {prob:.2f}")

        if pred == 1:
            st.error("⚠️ This person is **at risk for depression**.")
        else:
            st.success("✅ This person is **not at high risk for depression**.")

        # ---- Risk Gauge Visualization ----
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Depression Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, threshold*100], "color": "lightgreen"},
                    {"range": [threshold*100, 100], "color": "red"},
                ],
            }
        ))
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error processing input: {e}")