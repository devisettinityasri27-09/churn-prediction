import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def train_and_track(
    run_name="RandomForest_Baseline",
    params=None
):

    # ==========================================
    # MODEL PARAMETERS
    # ==========================================

    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    print("\n==========================================")
    print(f"Starting MLflow Run: {run_name}")
    print("==========================================")

    # ==========================================
    # LOAD PROCESSED DATA
    # ==========================================

    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    print("Training data shape:", X_train.shape)
    print("Testing data shape :", X_test.shape)

    # ==========================================
    # CREATE / SELECT MLFLOW EXPERIMENT
    # ==========================================

    mlflow.set_experiment(
        "Telco_Churn_Prediction"
    )

    # ==========================================
    # START MLFLOW RUN
    # ==========================================

    with mlflow.start_run(
        run_name=run_name
    ):

        # ======================================
        # LOG PARAMETERS
        # ======================================

        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForest"
        )

        # ======================================
        # CREATE MODEL
        # ======================================

        model = RandomForestClassifier(
            **params
        )

        # ======================================
        # TRAIN MODEL
        # ======================================

        print("\nTraining Random Forest...")

        model.fit(
            X_train,
            y_train
        )

        print("Model training completed!")

        # ======================================
        # PREDICTIONS
        # ======================================

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # ======================================
        # CALCULATE METRICS
        # ======================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        # ======================================
        # LOG METRICS TO MLFLOW
        # ======================================

        mlflow.log_metrics(
            metrics
        )

        print("\n==========================================")
        print("MODEL PERFORMANCE")
        print("==========================================")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        # ======================================
        # CREATE ARTIFACT DIRECTORY
        # ======================================

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # ======================================
        # CONFUSION MATRIX
        # ======================================

        print("\nCreating confusion matrix...")

        fig_cm, ax_cm = plt.subplots(
            figsize=(6, 5)
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax_cm,
            cmap="Blues"
        )

        ax_cm.set_title(
            f"Confusion Matrix - {run_name}"
        )

        cm_path = (
            "artifacts/confusion_matrix.png"
        )

        fig_cm.savefig(
            cm_path,
            bbox_inches="tight"
        )

        plt.close(fig_cm)

        # Log confusion matrix
        mlflow.log_artifact(
            cm_path,
            artifact_path="plots"
        )

        print(
            "Confusion matrix logged successfully!"
        )

        # ======================================
        # ROC CURVE
        # ======================================

        print("Creating ROC curve...")

        fig_roc, ax_roc = plt.subplots(
            figsize=(6, 5)
        )

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax_roc
        )

        ax_roc.set_title(
            f"ROC Curve - {run_name}"
        )

        roc_path = (
            "artifacts/roc_curve.png"
        )

        fig_roc.savefig(
            roc_path,
            bbox_inches="tight"
        )

        plt.close(fig_roc)

        # Log ROC curve
        mlflow.log_artifact(
            roc_path,
            artifact_path="plots"
        )

        print(
            "ROC curve logged successfully!"
        )

        # ======================================
        # LOG DATASET METADATA
        # ======================================

        metadata_path = (
            "data/processed/dataset_metadata.json"
        )

        if os.path.exists(metadata_path):

            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

            print(
                "Dataset metadata logged successfully!"
            )

        # ======================================
        # LOG MODEL TO MLFLOW
        # ======================================

        print("\nLogging model to MLflow...")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print(
            "Model logged successfully!"
        )

        # ======================================
        # SAVE LOCAL MODEL BACKUP
        # ======================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/random_forest_model.pkl"
        )

        print(
            "Local model saved to:"
        )

        print(
            "models/random_forest_model.pkl"
        )

        # ======================================
        # SUCCESS
        # ======================================

        print("\n==========================================")
        print("MLFLOW RUN COMPLETED SUCCESSFULLY!")
        print("==========================================")

        print(
            f"Run Name : {run_name}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train_and_track(
        run_name="RandomForest"
    )