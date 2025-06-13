import os 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from io import BytesIO

# Load model and scaler
def load_model():
    # Get the path of the current script (Deployment/Student.py)
    model_path = os.path.join(os.path.dirname(__file__), 'student_performance_model.pkl')
    
    # Load the model and scaler from the relative path
    with open(model_path, 'rb') as file:
        model, scaler = pickle.load(file)[:2]
    
    return model, scale

# Preprocess input for single prediction
def preprocess_input(data_dict, scaler):
    mapping = {"Yes": 1, "No": 0}
    data_dict["Extracurricular Activities"] = mapping[data_dict["Extracurricular Activities"]]
    df = pd.DataFrame([data_dict])
    df_scaled = scaler.transform(df)
    return df_scaled, df

# Preprocess input for batch prediction
def preprocess_batch(df, scaler):
    df = df.copy()
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    df_scaled = scaler.transform(df)
    return df_scaled, df

# Predict
def predict(model, X):
    return model.predict(X)

# SHAP
def shap_explain(model, X_scaled):
    explainer = shap.Explainer(model.predict, X_scaled)
    shap_values = explainer(X_scaled)
    st.subheader("📈 SHAP Waterfall Plot")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=5, show=False)
    st.pyplot(fig)

# LIME
def lime_explain(model, X_scaled, df_original):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(df_original),
        feature_names=df_original.columns,
        mode="regression"
    )
    exp = explainer.explain_instance(X_scaled[0], model.predict, num_features=5)
    st.subheader("🧪 LIME Explanation")
    st.components.v1.html(exp.as_html(), height=400)

# Convert to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Prediction Visualization
def visualize_predictions(df):
    st.subheader("📊 Prediction Distribution")

    st.bar_chart(df[["Predicted Performance"]])
    
    fig, ax = plt.subplots()
    ax.hist(df["Predicted Performance"], bins=10, color="skyblue", edgecolor="black")
    ax.set_xlabel("Performance Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Predicted Scores")
    st.pyplot(fig)

# Main Streamlit App
def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="centered")
    st.title("🎓 Student Performance Prediction App")

    model, scaler = load_model()

    mode = st.sidebar.radio("Select Mode", ["🔍 Single Prediction", "📁 Bulk Prediction (CSV)"])

    if mode == "🔍 Single Prediction":
        st.subheader("🔍 Predict for One Student")

        hours_studied = st.number_input("Hours Studied", 0, 10, 5)
        previous_scores = st.number_input("Previous Scores (%)", 0, 100, 60)
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        sleep_hours = st.number_input("Sleep Hours", 0, 12, 6)
        papers_practiced = st.number_input("Sample Question Papers Practiced", 0, 10, 3)

        input_data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extracurricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": papers_practiced
        }

        show_shap = st.checkbox("📈 Show SHAP Explanation")
        show_lime = st.checkbox("🧪 Show LIME Explanation")

        if st.button("🎯 Predict"):
            try:
                X_scaled, df_original = preprocess_input(input_data, scaler)
                result = predict(model, X_scaled)
                st.success(f"🎯 Predicted Performance Index: {result[0]:.2f}")
                if show_shap:
                    shap_explain(model, X_scaled)
                if show_lime:
                    lime_explain(model, X_scaled, df_original)
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

    elif mode == "📁 Bulk Prediction (CSV)":
        st.subheader("📁 Upload CSV File for Bulk Prediction")

        st.markdown("""
        **📌 Required CSV Columns:**
        - Hours Studied
        - Previous Scores
        - Extracurricular Activities (Yes/No)
        - Sleep Hours
        - Sample Question Papers Practiced
        """)

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                X_scaled, df_processed = preprocess_batch(df, scaler)
                predictions = predict(model, X_scaled)
                df_processed["Predicted Performance"] = predictions

                st.success("✅ Predictions Completed!")
                st.dataframe(df_processed)

                visualize_predictions(df_processed)

                csv = convert_df_to_csv(df_processed)
                st.download_button("⬇️ Download Predictions as CSV", csv, "predicted_results.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

# Run App
if __name__ == "__main__":
    main()
