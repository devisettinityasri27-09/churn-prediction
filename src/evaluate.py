import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


def evaluate_model(model, name, X_test, y_test):

    # Generate predictions
    y_pred = model.predict(X_test)

    # Probability of Churn = 1
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": roc_auc,
        "fpr": fpr,
        "tpr": tpr
    }


def evaluate_all_models():

    # ==========================================
    # LOAD TEST DATA
    # ==========================================

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    print("Test data loaded successfully!")
    print("Test data shape:", X_test.shape)

    # ==========================================
    # LOAD TRAINED MODELS
    # ==========================================

    model_files = {
        "Logistic Regression":
            "models/logistic_regression_baseline.pkl",

        "Decision Tree":
            "models/decision_tree_baseline.pkl",

        "Random Forest":
            "models/random_forest_baseline.pkl"
    }

    results = {}
    roc_data = {}

    # ==========================================
    # EVALUATE EACH MODEL
    # ==========================================

    print("\n==========================================")
    print("MODEL EVALUATION")
    print("==========================================\n")

    for model_name, model_path in model_files.items():

        model = joblib.load(model_path)

        result = evaluate_model(
            model,
            model_name,
            X_test,
            y_test
        )

        results[model_name] = result

        roc_data[model_name] = {
            "fpr": result["fpr"],
            "tpr": result["tpr"],
            "auc": result["AUC"]
        }

        print(f"--- {model_name} ---")
        print(f"Accuracy : {result['Accuracy']:.4f}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall   : {result['Recall']:.4f}")
        print(f"F1-Score : {result['F1']:.4f}")
        print(f"ROC-AUC  : {result['AUC']:.4f}")
        print()

    # ==========================================
    # SELECT BEST MODEL BY RECALL
    # ==========================================

    best_model_name = max(
        results,
        key=lambda name: results[name]["Recall"]
    )

    best_model = joblib.load(
        model_files[best_model_name]
    )

    best_recall = results[
        best_model_name
    ]["Recall"]

    print("==========================================")
    print(
        f"Best Model: {best_model_name} "
        f"(Recall: {best_recall:.4f})"
    )
    print("==========================================")

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    results_for_csv = {}

    for name, result in results.items():

        results_for_csv[name] = {
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1": result["F1"],
            "ROC-AUC": result["AUC"]
        }

    results_df = pd.DataFrame(
        results_for_csv
    ).T

    results_df.to_csv(
        "outputs/baseline_model_results.csv"
    )

    print("\nBaseline results saved successfully!")

    # ==========================================
    # ERROR ANALYSIS
    # ==========================================

    predictions = best_model.predict(X_test)

    errors_df = pd.DataFrame({
        "Actual_Churn": y_test,
        "Predicted_Churn": predictions
    })

    false_negatives = errors_df[
        (errors_df["Actual_Churn"] == 1) &
        (errors_df["Predicted_Churn"] == 0)
    ]

    false_positives = errors_df[
        (errors_df["Actual_Churn"] == 0) &
        (errors_df["Predicted_Churn"] == 1)
    ]

    false_negatives.to_csv(
        "outputs/false_negatives.csv",
        index=False
    )

    false_positives.to_csv(
        "outputs/false_positives.csv",
        index=False
    )

    print("False negatives saved successfully!")
    print("False positives saved successfully!")

    print("\nFalse Negatives:", len(false_negatives))
    print("False Positives:", len(false_positives))


if __name__ == "__main__":
    evaluate_all_models()