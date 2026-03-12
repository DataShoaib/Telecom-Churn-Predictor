import pickle
import pandas as pd
import shap
import numpy as np
from openai import OpenAI
from sklearn.pipeline import Pipeline
from logger.logger import get_logger
from dotenv import load_dotenv
import os
import streamlit as st   # ✅ ADDED

load_dotenv()

logger = get_logger("summary-generator")

# ==============================
# API KEY LOADING (UPDATED)
# ==============================

# First try Streamlit secrets, then fallback to .env
api_key = st.secrets.get("OPENAI_API_KEY") 

if not api_key:
    raise ValueError("API key not found. Add it to Streamlit Secrets or .env")
else:
    logger.info("API key loaded successfully!")

# ==============================


def load_model(model_path: str) -> Pipeline:
    try:
        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


best_model = load_model("models/model.pkl")
preprocessor = best_model.named_steps["preprocessor"]
model = best_model.named_steps["model"]
x_test = load_data("data/processed/split/X_test.csv")
x_test_transformed = preprocessor.transform(x_test)


# Get real feature names after preprocessing
def get_feature_names(preprocessor, input_df):
    try:
        names = preprocessor.get_feature_names_out()
        names = [n.split("__")[-1] for n in names]
        return names
    except Exception:
        pass

    try:
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(cols)
                feature_names.extend([n.split("__")[-1] for n in names])
            else:
                feature_names.extend(cols)
        return feature_names
    except Exception:
        pass

    return [f"feature_{i}" for i in range(x_test_transformed.shape[1])]


feature_names = get_feature_names(preprocessor, x_test)

explainer = shap.LinearExplainer(model, x_test_transformed)

# ==============================
# GROQ CLIENT
# ==============================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# ==============================


def generate_customer_insights(customer):
    insights = []

    if customer.get("Contract") == "Month-to-month":
        insights.append(
            "Customer is on a month-to-month contract which often correlates with higher churn."
        )

    if customer.get("PaymentMethod") == "Electronic check":
        insights.append(
            "Electronic check payment users historically show higher churn behavior."
        )

    if customer.get("tenure", 0) < 12:
        insights.append(
            "Customer joined recently and early churn is common in the first year."
        )

    if customer.get("tenure", 0) > 48:
        insights.append(
            "Long tenure customers are usually more loyal and less likely to churn."
        )

    return insights


def predict_and_explain(customer_data: dict):

    customer_df = pd.DataFrame([customer_data])

    churn_prob = best_model.predict_proba(customer_df)[0][1]

    churn_pred = "High Risk 🔴" if churn_prob > 0.5 else "Low Risk 🟢"

    transformed = preprocessor.transform(customer_df)

    if hasattr(transformed, "toarray"):
        transformed_dense = transformed.toarray()
    else:
        transformed_dense = np.array(transformed)

    shap_vals = explainer.shap_values(transformed_dense)[0]

    data_row = transformed_dense[0]

    feature_shap = list(zip(feature_names, shap_vals))

    risk_factors = sorted(
        feature_shap,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    risk_text = "\n".join(
        [f"{i+1}. {name} (impact {value:.3f})"
         for i, (name, value) in enumerate(risk_factors)]
    )

    insights = generate_customer_insights(customer_data)

    insight_text = "\n".join([f"- {i}" for i in insights])

    prompt = f"""
You are a telecom customer retention expert.

Customer churn probability: {churn_prob:.2%}

Customer insights:
{insight_text}

Top churn drivers detected by the ML model:
{risk_text}

Explain the churn risk in simple business language.

Return sections:

CUSTOMER RISK SUMMARY

KEY RISK DRIVERS

RECOMMENDED RETENTION ACTIONS
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )

    ai_text = response.choices[0].message.content

    shap_explanation = shap.Explanation(
        values=shap_vals,
        base_values=float(explainer.expected_value),
        data=data_row,
        feature_names=feature_names
    )

    return {
        "churn_probability": f"{churn_prob:.2%}",
        "risk_level": churn_pred,
        "top_risk_factors": [f[0] for f in risk_factors],
        "ai_summary": ai_text,
        "shap_values": shap_explanation
    }