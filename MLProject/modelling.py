
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import os
from mlflow.models.signature import infer_signature

# --- Setup DagsHub (advanced) ---
# dagshub.init(repo_owner='zaamuhammad711', repo_name='stroke-mlflow', mlflow=True)

# Load dataset
data_path = "healthcare-dataset-stroke-data_preprocessing/healthcare-dataset-stroke-data_preprocessing.csv"
df = pd.read_csv(data_path)

# Pisahkan fitur dan target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Mulai MLflow run ---
with mlflow.start_run():

    # Hyperparameter tuning RandomForest
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    # Prediksi & hitung akurasi
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")

    # --- Logging ke MLflow / DagsHub ---
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)

    # Tentukan signature untuk model agar tidak ada warning
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        grid.best_estimator_,
        "model",
        signature=signature,
        input_example=X_test.iloc[:5]
    )

    # --- Logging artefak tambahan (minimal 2 untuk advanced) ---
    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # 2. Sample test dataset
    test_sample_path = "test_sample.csv"
    X_test_sample = X_test.copy()
    X_test_sample["y_true"] = y_test
    X_test_sample["y_pred"] = y_pred
    X_test_sample.to_csv(test_sample_path, index=False)
    mlflow.log_artifact(test_sample_path)

    # --- Optional: Hapus file lokal setelah logging ---
    os.remove(cm_path)
    os.remove(test_sample_path)
