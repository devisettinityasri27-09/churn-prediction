import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models():

    # ==========================================
    # LOAD PROCESSED TRAINING DATA
    # ==========================================

    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    print("Processed training data loaded successfully!")
    print("Training data shape:", X_train.shape)
    print("Training target shape:", y_train.shape)

    # ==========================================
    # DEFINE BASELINE MODELS
    # ==========================================

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
    }

    # ==========================================
    # TRAIN MODELS
    # ==========================================

    trained_models = {}

    print("\n==========================================")
    print("MODEL TRAINING")
    print("==========================================")

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(
            X_train,
            y_train
        )

        trained_models[model_name] = model

        print(f"{model_name} trained successfully!")

    # ==========================================
    # CREATE MODELS DIRECTORY
    # ==========================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ==========================================
    # SAVE TRAINED MODELS
    # ==========================================

    print("\n==========================================")
    print("SAVING MODELS")
    print("==========================================")

    for model_name, model in trained_models.items():

        file_name = (
            model_name
            .replace(" ", "_")
            .lower()
        )

        model_path = (
            f"models/{file_name}_baseline.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"{model_name} saved to: {model_path}"
        )

    print("\nAll baseline models trained and saved successfully!")


# ==========================================
# RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    train_models()