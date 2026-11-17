import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.joblib"
FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter passenger details to estimate survival using a model "
    "trained on historical Titanic data."
)

st.divider()

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Gender", ["female", "male"], horizontal=True)
age = st.slider("Age", min_value=1, max_value=80, value=30)
fare = st.slider("Fare Paid", min_value=0.0, max_value=500.0, value=30.0)

if st.button("Predict Survival", type="primary"):
    try:
        model = load_model()
        passenger = pd.DataFrame(
            [[pclass, 1 if sex == "female" else 0, age, fare]],
            columns=FEATURES
        )

        prediction = int(model.predict(passenger)[0])
        probability = float(model.predict_proba(passenger)[0][1])

        st.subheader("Prediction Result")
        st.metric("Estimated Survival Probability", f"{probability:.1%}")

        if prediction == 1:
            st.success("The model predicts survival.")
        else:
            st.error("The model predicts non-survival.")

    except FileNotFoundError:
        st.error(
            "model.joblib was not found. Add your trained model file "
            "to this same project folder."
        )
    except Exception as error:
        st.error(f"Prediction error: {error}")

st.divider()
st.caption(
    "Educational demonstration using historical Titanic data. "
    "Predictions are estimates, not real-world advice."
)
