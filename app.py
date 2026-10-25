from pathlib import Path
import json, sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from ml_backend import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

st.set_page_config(page_title="Solar PV Fault Classification",page_icon="☀️",layout="wide")

@st.cache_resource
def load_model():
    model_path=ROOT/"models"/"knn_pipeline.joblib"
    metadata_path=ROOT/"models"/"model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifact missing. Run `python train_model.py` first.")
    return joblib.load(model_path),json.loads(metadata_path.read_text(encoding="utf-8"))

st.title("☀️ Solar Panel Fault Classification")
st.caption("K-Nearest Neighbors (KNN) · Jupyter Notebook backend · Streamlit GUI")
try:
    model,metadata=load_model()
except Exception as e:
    st.error(str(e)); st.stop()

pages=["Dashboard","Single Prediction","Batch CSV Prediction","Model Evaluation","Dataset Explorer","About"]
page=st.sidebar.radio("Navigate",pages)
metrics=metadata.get("test_metrics",{})

if page=="Dashboard":
    st.subheader("Held-out test performance")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Accuracy",f'{metrics.get("accuracy",0):.2%}')
    c2.metric("Macro F1",f'{metrics.get("f1_macro",0):.2%}')
    c3.metric("Balanced accuracy",f'{metrics.get("balanced_accuracy",0):.2%}')
    c4.metric("MCC",f'{metrics.get("mcc",0):.3f}')
    st.info("Educational demonstration using a synthetic, engineering-inspired dataset. These scores do not establish performance on real PV installations.")
    st.write("**Classes:**",", ".join(metadata.get("classes",[])))
    st.write("**Selected KNN parameters:**",metadata.get("best_parameters",{}))
    st.write(f"Training samples: {metadata.get('n_train','N/A'):,} · Test samples: {metadata.get('n_test','N/A'):,}")

elif page=="Single Prediction":
    st.subheader("Classify a new PV operating point")
    st.write("Enter values in the units indicated. Defaults are illustrative only.")
    with st.form("single_prediction"):
        defaults={"irradiance_W_m2":700.0,"module_temp_C":42.0,"dc_voltage_V":360.0,
                  "dc_current_A":8.0,"dc_power_W":2800.0,"fill_factor":0.75,
                  "insulation_resistance_MOhm":10.0,"voltage_ripple_V":1.5,
                  "current_imbalance_pct":2.0,"temperature_rise_C":4.0}
        values={}
        left,right=st.columns(2)
        for i,f in enumerate(NUMERIC_FEATURES):
            with (left if i%2==0 else right):
                values[f]=st.number_input(f,value=float(defaults[f]),format="%.4f")
        values["operating_mode"]=st.selectbox("Operating mode",["MPPT","FixedVoltage","Diagnostic"])
        submitted=st.form_submit_button("Predict fault class",type="primary")
    if submitted:
        X=pd.DataFrame([values],columns=FEATURES)
        pred=str(model.predict(X)[0])
        st.subheader("Prediction")
        st.success(f"Predicted operating state: **{pred}**")
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba(X)[0]
            probdf=pd.DataFrame({"Class":model.classes_,"KNN vote probability":probs}).sort_values("KNN vote probability",ascending=False)
            st.dataframe(probdf,use_container_width=True)
            st.bar_chart(probdf.set_index("Class"))
            st.caption("KNN vote proportions are not calibrated real-world fault probabilities.")

elif page=="Batch CSV Prediction":
    st.subheader("Upload measurements for batch classification")
    st.write("CSV must contain the 10 numeric feature columns and `operating_mode`. Target column is optional.")
    up=st.file_uploader("Upload CSV",type=["csv"])
    if up:
        try:
            data=pd.read_csv(up)
            missing=[f for f in FEATURES if f not in data.columns]
            if missing:
                st.error("Missing required columns: "+", ".join(missing))
            else:
                X=data[FEATURES].copy()
                for f in NUMERIC_FEATURES:
                    X[f]=pd.to_numeric(X[f],errors="coerce")
                preds=model.predict(X)
                out=data.copy()
                out["predicted_fault"]=preds
                if hasattr(model,"predict_proba"):
                    out["knn_vote_score"]=model.predict_proba(X).max(axis=1)
                st.success(f"Predicted {len(out):,} rows.")
                st.dataframe(out.head(200),use_container_width=True)
                st.download_button("Download predictions CSV",out.to_csv(index=False).encode("utf-8"),
                                   file_name="solar_fault_predictions.csv",mime="text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")

elif page=="Model Evaluation":
    st.subheader("Test metrics")
    st.dataframe(pd.DataFrame([{"Metric":k,"Score":v} for k,v in metrics.items()]),use_container_width=True)
    for name,title in [("confusion_matrix.png","Confusion Matrix"),("per_class_metrics.png","Per-class Metrics"),
                       ("class_distribution.png","Class Distribution"),("feature_correlation.png","Feature Correlation"),
                       ("feature_selection.png","Feature Selection"),("missing_values.png","Missing Values")]:
        p=ROOT/"reports"/"figures"/name
        if p.exists():
            st.subheader(title); st.image(str(p),use_container_width=True)
    st.write("Detailed classification report:")
    report=metadata.get("classification_report",{})
    if report:
        st.dataframe(pd.DataFrame(report).T,use_container_width=True)

elif page=="Dataset Explorer":
    data_path=ROOT/"data"/"solar_pv_faults_synthetic.csv"
    data=pd.read_csv(data_path)
    st.subheader("Dataset preview")
    st.write(f"Rows: {len(data):,} · Columns: {len(data.columns)}")
    st.dataframe(data.head(100),use_container_width=True)
    st.subheader("Class counts")
    st.bar_chart(data["fault_label"].value_counts())
    st.download_button("Download supplied dataset",data_path.read_bytes(),
                       file_name=data_path.name,mime="text/csv")

else:
    st.subheader("About this project")
    st.markdown("""
    **Algorithm:** K-Nearest Neighbors (KNN), selected using stratified 5-fold cross-validation.

    **Preprocessing:** exact duplicate removal, numeric median imputation, training-fold IQR clipping,
    standardization, categorical most-frequent imputation, one-hot encoding, and mutual-information
    feature selection.

    **Limitations:** The dataset is synthetic and designed for coursework demonstrations. It does not
    represent measured field data and must not be used for safety-critical decisions. Validate any
    real-world model on independent measurements from the target installation.
    """)
