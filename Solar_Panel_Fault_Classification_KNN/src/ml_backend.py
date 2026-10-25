"""Reusable KNN training, preprocessing, evaluation and prediction utilities."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix)

TARGET = "fault_label"
NUMERIC_FEATURES = [
    "irradiance_W_m2", "module_temp_C", "dc_voltage_V", "dc_current_A",
    "dc_power_W", "fill_factor", "insulation_resistance_MOhm",
    "voltage_ripple_V", "current_imbalance_pct", "temperature_rise_C"
]
CATEGORICAL_FEATURES = ["operating_mode"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to training-fold IQR fences (outlier mitigation)."""
    def __init__(self, factor=1.5):
        self.factor = factor
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * iqr
        self.upper_ = self.q3_ + self.factor * iqr
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)
    def get_feature_names_out(self, input_features=None):
        """Preserve column names because this transformer does not change columns."""
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(len(self.lower_))], dtype=object)
        return np.asarray(input_features, dtype=object)

def clean_dataframe(df):
    """Drop exact duplicate records and rows without a known target."""
    df = df.copy()
    df = df.dropna(subset=[TARGET])
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"rows_before": int(before), "rows_after": int(len(df)),
                "duplicates_removed": int(before-len(df))}

def build_pipeline(numeric_features=None, categorical_features=None):
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_clip", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ], remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectKBest(score_func=mutual_info_classif, k="all")),
        ("knn", KNeighborsClassifier(n_jobs=-1))
    ])

def train_and_evaluate(data_path, model_path, metadata_path, figure_dir,
                       test_size=0.20, random_state=42, run_grid_search=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_path, model_path, metadata_path, figure_dir = map(Path, [data_path, model_path, metadata_path, figure_dir])
    df = pd.read_csv(data_path)
    df, cleaning = clean_dataframe(df)
    missing = [c for c in FEATURES+[TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(str)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].replace([np.inf,-np.inf], np.nan)
    # Ensure categorical values remain strings and missing values can be imputed.
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("object")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipe = build_pipeline()
    # Preprocessed feature count: 10 numeric + 3 mode categories = 13.
    param_grid = {
        "select__k": ["all", 10],
        "knn__n_neighbors": [5, 7, 9],
        "knn__weights": ["distance"],
        "knn__p": [1, 2]
    }
    if run_grid_search:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=1,
                              verbose=0, refit=True, return_train_score=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = float(search.best_score_)
        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cv_results.to_csv(figure_dir.parent/"cv_results.csv", index=False)
    else:
        model = pipe.set_params(select__k="all", knn__n_neighbors=7,
                                knn__weights="distance", knn__p=2)
        model.fit(X_train, y_train)
        best_params = {"select__k":"all","knn__n_neighbors":7,"knn__weights":"distance","knn__p":2}
        cv_score = None

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test,y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test,y_pred)),
        "precision_macro": float(precision_score(y_test,y_pred,average="macro",zero_division=0)),
        "recall_macro": float(recall_score(y_test,y_pred,average="macro",zero_division=0)),
        "f1_macro": float(f1_score(y_test,y_pred,average="macro",zero_division=0)),
        "f1_weighted": float(f1_score(y_test,y_pred,average="weighted",zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test,y_pred))
    }
    report = classification_report(y_test,y_pred,output_dict=True,zero_division=0)
    report_df = pd.DataFrame(report).T
    figure_dir.mkdir(parents=True,exist_ok=True)
    # Class distribution
    plt.figure(figsize=(10,5))
    df[TARGET].value_counts().plot(kind="bar")
    plt.title("Synthetic Solar PV Fault Dataset: Class Distribution")
    plt.xlabel("Fault class"); plt.ylabel("Count")
    plt.xticks(rotation=25,ha="right"); plt.tight_layout()
    plt.savefig(figure_dir/"class_distribution.png",dpi=150); plt.close()
    # Missing values
    plt.figure(figsize=(11,4))
    sns.heatmap(df[FEATURES].isna().sample(min(500,len(df)),random_state=random_state),cbar=False)
    plt.title("Missing-Value Pattern (sample)"); plt.tight_layout()
    plt.savefig(figure_dir/"missing_values.png",dpi=150); plt.close()
    # Correlation
    plt.figure(figsize=(11,8))
    sns.heatmap(df[NUMERIC_FEATURES].corr(),cmap="coolwarm",center=0)
    plt.title("Numerical Feature Correlation"); plt.tight_layout()
    plt.savefig(figure_dir/"feature_correlation.png",dpi=150); plt.close()
    # Confusion matrix
    labels=list(model.classes_)
    cm=confusion_matrix(y_test,y_pred,labels=labels)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm,annot=True,fmt="d",xticklabels=labels,yticklabels=labels,cmap="Blues")
    plt.title("KNN Confusion Matrix (held-out test set)")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.xticks(rotation=25,ha="right")
    plt.tight_layout(); plt.savefig(figure_dir/"confusion_matrix.png",dpi=150); plt.close()
    # Per-class metrics
    cls=report_df.loc[[x for x in labels if x in report_df.index],["precision","recall","f1-score"]]
    cls.plot(kind="bar",figsize=(11,5)); plt.title("Per-Class Performance")
    plt.ylim(0,1.05); plt.xticks(rotation=25,ha="right"); plt.tight_layout()
    plt.savefig(figure_dir/"per_class_metrics.png",dpi=150); plt.close()
    # Feature selector scores from transformed matrix
    try:
        names=model.named_steps["preprocess"].get_feature_names_out()
        scores=model.named_steps["select"].scores_
        score_df=pd.DataFrame({"feature":names,"mutual_information":scores}).sort_values("mutual_information",ascending=False)
        score_df.to_csv(figure_dir.parent/"feature_scores.csv",index=False)
        plt.figure(figsize=(10,6))
        top=score_df.head(15).sort_values("mutual_information")
        plt.barh(top["feature"],top["mutual_information"])
        plt.title("Mutual Information Feature Scores"); plt.tight_layout()
        plt.savefig(figure_dir/"feature_selection.png",dpi=150); plt.close()
    except Exception:
        score_df=pd.DataFrame()

    model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,model_path)
    metadata = {
        "project":"Solar Panel Fault Classification using KNN",
        "dataset_type":"synthetic educational dataset; not field measurements",
        "target_column":TARGET,"input_features":FEATURES,
        "numeric_features":NUMERIC_FEATURES,"categorical_features":CATEGORICAL_FEATURES,
        "classes":[str(c) for c in model.classes_],
        "best_parameters":best_params,"cv_macro_f1":cv_score,
        "test_metrics":metrics,"test_size":test_size,"random_state":random_state,
        "cleaning":cleaning,"n_samples_after_duplicate_removal":int(len(df)),
        "n_train":int(len(X_train)),"n_test":int(len(X_test)),
        "classification_report":report
    }
    metadata_path.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    return model, metadata, report_df
