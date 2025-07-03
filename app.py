#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import gradio as gr
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


# Load model
model = joblib.load("models/xgb_market_mood.pkl")


# In[3]:


# Feature names
features = [
    "Mood_Index", "Mood_Index_Lag1",
    "SP500_Returns_Lag1", "VIX_Change_Lag1",
    "Google_Trend_Lag1", "Unemployment_Lag1"
]


# In[4]:


# SHAP Explainer setup
sample_X = pd.DataFrame([np.zeros(len(features))], columns=features)
explainer = shap.Explainer(model, sample_X)


# In[5]:


# Prediction function
def predict_drop_risk(
    Mood_Index, Mood_Index_Lag1, SP500_Returns_Lag1,
    VIX_Change_Lag1, Google_Trend_Lag1, Unemployment_Lag1
):
    input_df = pd.DataFrame([[Mood_Index, Mood_Index_Lag1,
                              SP500_Returns_Lag1, VIX_Change_Lag1,
                              Google_Trend_Lag1, Unemployment_Lag1]],
                            columns=features)

    prob = model.predict_proba(input_df)[:, 1][0]

    shap_values = explainer(input_df)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()

    output_path = "images/explain/shap_waterfall_user.png"
    full_save_path = os.path.join(os.getcwd(), output_path)

    plt.savefig(full_save_path, bbox_inches="tight", facecolor="white")
    plt.close()

    return f"📈 Probability of market drop next week: {prob:.2%}", output_path


# In[6]:


# Gradio app
outputs = [
    gr.Textbox(label="Probability"),
    gr.Image(label="SHAP Waterfall Plot")
]

inputs = [
    gr.Number(label="Mood_Index"),
    gr.Number(label="Mood_Index_Lag1"),
    gr.Number(label="SP500_Returns_Lag1"),
    gr.Number(label="VIX_Change_Lag1"),
    gr.Number(label="Google_Trend_Lag1"),
    gr.Number(label="Unemployment_Lag1")
]

gr.Interface(
    fn=predict_drop_risk,
    inputs=inputs,
    outputs=outputs,
    title="Market Mood Forecasting",
    description="Predicts probability of market drop next week with SHAP explainability."
).launch()

