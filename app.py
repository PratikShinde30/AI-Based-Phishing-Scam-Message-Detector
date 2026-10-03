
import streamlit as st
import joblib
import json

from pathlib import Path

# ----------------------------------------
# Page configuration
# ----------------------------------------
st.set_page_config(
    page_title="AI Scam Message Detector",
    page_icon="🛡️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_detector.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


# ----------------------------------------
# Load trained model
# ----------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ----------------------------------------
# Application interface
# ----------------------------------------
st.title("🛡️ AI-Based Phishing/Scam Message Detector")

st.markdown(
    """
    Enter an SMS or text message below to check whether it resembles
    legitimate or spam messages in the training dataset.

    **Note:** This tool is a spam classifier, not a guarantee that a
    message or link is safe.
    """
)

st.divider()

# Check if model has been trained
if not MODEL_PATH.exists():
    st.error("The trained model was not found.")
    st.info(
        "Open the VS Code terminal and run: "
        "`python train_model.py`"
    )
    st.stop()

model = load_model()

# Message input
message = st.text_area(
    "Enter your message",
    placeholder=(
        "Example: Congratulations! You have won a prize. "
        "Click the link to claim it."
    ),
    height=150,
    max_chars=10000
)

# Prediction
if st.button("🔍 Detect Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message before clicking Detect.")
    else:
        prediction = int(model.predict([message])[0])

        # Get class probabilities
        probabilities = model.predict_proba([message])[0]
        class_labels = list(model.classes_)

        spam_index = class_labels.index(1)
        spam_probability = float(probabilities[spam_index])

        st.divider()
        st.subheader("Detection Result")

        if prediction == 1:
            st.error("🚨 Spam / Suspicious Message")
            st.write(
                "This message resembles spam messages in the "
                "training dataset. Avoid clicking suspicious links "
                "or sharing personal information."
            )
        else:
            st.success("✅ Likely Legitimate Message")
            st.write(
                "The model classified this message as legitimate. "
                "This does not guarantee that it is safe."
            )

        st.metric(
            "Model's estimated spam probability",
            f"{spam_probability * 100:.2f}%"
        )

        st.progress(spam_probability)

        st.caption(
            "This probability is based on the trained model and "
            "is not a verified measure of real-world scam risk."
        )

# ----------------------------------------
# Model performance section
# ----------------------------------------
st.divider()
st.subheader("📊 Model Performance")

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")

    col3, col4 = st.columns(2)

    col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    col4.metric("F1-score", f"{metrics['f1_score'] * 100:.2f}%")

    st.caption(
        f"Dataset: {metrics['total_samples']} messages | "
        f"Training samples: {metrics['training_samples']} | "
        f"Testing samples: {metrics['testing_samples']}"
    )
else:
    st.info("Train the model to view evaluation metrics.")

# ----------------------------------------
# Footer
# ----------------------------------------
st.divider()
st.caption(
    "Academic AIML Project | TF-IDF + Multinomial Naive Bayes"
)