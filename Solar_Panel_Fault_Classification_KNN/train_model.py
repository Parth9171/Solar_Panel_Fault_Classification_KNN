from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from ml_backend import train_and_evaluate

model, metadata, report = train_and_evaluate(
    data_path=ROOT/"data"/"solar_pv_faults_synthetic.csv",
    model_path=ROOT/"models"/"knn_pipeline.joblib",
    metadata_path=ROOT/"models"/"model_metadata.json",
    figure_dir=ROOT/"reports"/"figures",
    run_grid_search=True
)
print("Best parameters:",metadata["best_parameters"])
print("CV macro F1:",metadata["cv_macro_f1"])
print("Test metrics:")
for k,v in metadata["test_metrics"].items():
    print(f"  {k}: {v:.4f}")
print("Model saved to models/knn_pipeline.joblib")
