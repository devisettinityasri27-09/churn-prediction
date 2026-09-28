import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def validate_reproducibility():

    print("\n==========================================")
    print("REPRODUCIBILITY VALIDATION")
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

    print("Data loaded successfully!")
    print("Training shape:", X_train.shape)
    print("Testing shape :", X_test.shape)

    # ==========================================
    # FIXED MODEL PARAMETERS
    # ==========================================

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced"
    }

    # ==========================================
    # TRAIN MODEL - RUN 1
    # ==========================================

    print("\nTraining model - Run 1...")

    model_1 = RandomForestClassifier(
        **params
    )

    model_1.fit(
        X_train,
        y_train
    )

    predictions_1 = model_1.predict(
        X_test
    )

    accuracy_1 = accuracy_score(
        y_test,
        predictions_1
    )

    print(
        f"Run 1 Accuracy: {accuracy_1:.4f}"
    )

    # ==========================================
    # TRAIN MODEL - RUN 2
    # ==========================================

    print("\nTraining model - Run 2...")

    model_2 = RandomForestClassifier(
        **params
    )

    model_2.fit(
        X_train,
        y_train
    )

    predictions_2 = model_2.predict(
        X_test
    )

    accuracy_2 = accuracy_score(
        y_test,
        predictions_2
    )

    print(
        f"Run 2 Accuracy: {accuracy_2:.4f}"
    )

    # ==========================================
    # COMPARE RESULTS
    # ==========================================

    print("\n==========================================")
    print("REPRODUCIBILITY RESULTS")
    print("==========================================")

    print(
        f"Run 1 Accuracy: {accuracy_1:.4f}"
    )

    print(
        f"Run 2 Accuracy: {accuracy_2:.4f}"
    )

    # Check predictions
    predictions_same = np.array_equal(
        predictions_1,
        predictions_2
    )

    accuracy_same = (
        accuracy_1 == accuracy_2
    )

    print(
        f"Predictions identical: {predictions_same}"
    )

    print(
        f"Accuracy identical: {accuracy_same}"
    )

    # ==========================================
    # FINAL VALIDATION
    # ==========================================

    if predictions_same and accuracy_same:

        print("\n==========================================")
        print("REPRODUCIBILITY VALIDATION PASSED!")
        print("==========================================")

        print(
            "The same data, parameters and random seed "
            "produced identical results."
        )

    else:

        print("\n==========================================")
        print("REPRODUCIBILITY VALIDATION FAILED!")
        print("==========================================")

        print(
            "The two runs produced different results."
        )

        raise RuntimeError(
            "Model reproducibility validation failed."
        )

    # ==========================================
    # SAVE VALIDATION RESULT
    # ==========================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    result = {
        "run_1_accuracy": accuracy_1,
        "run_2_accuracy": accuracy_2,
        "predictions_identical": predictions_same,
        "accuracy_identical": accuracy_same,
        "reproducibility_passed": (
            predictions_same and accuracy_same
        )
    }

    joblib.dump(
        result,
        "outputs/reproducibility_result.pkl"
    )

    print(
        "\nValidation result saved to:"
    )

    print(
        "outputs/reproducibility_result.pkl"
    )


if __name__ == "__main__":

    validate_reproducibility()