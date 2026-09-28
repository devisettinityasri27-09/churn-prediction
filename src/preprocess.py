import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_data():

    # Load raw dataset
    data_path = "data/raw/Churn.csv"
    df = pd.read_csv(data_path)

    print("Dataset loaded successfully!")
    print("Original shape:", df.shape)

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    # Drop customer ID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Encode target
    df["Churn"] = df["Churn"].apply(
        lambda x: 1 if str(x).strip().lower() == "yes" else 0
    )

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Identify column types
    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns

    num_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Scale numerical features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train[num_cols]
    )

    X_test_scaled = scaler.transform(
        X_test[num_cols]
    )

    # Encode categorical features
    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    X_train_encoded = ohe.fit_transform(
        X_train[cat_cols]
    )

    X_test_encoded = ohe.transform(
        X_test[cat_cols]
    )

    # Combine numerical and categorical features
    X_train_final = np.hstack(
        (X_train_scaled, X_train_encoded)
    )

    X_test_final = np.hstack(
        (X_test_scaled, X_test_encoded)
    )

    # Create directories if required
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save processed arrays
    np.save(
        "data/processed/X_train_final.npy",
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.values
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.values
    )

    # Save preprocessing objects
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        ohe,
        "models/ohe.pkl"
    )

    # Save metadata
    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": X_train_final.shape,
        "test_shape": X_test_final.shape,
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols)
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")
    print("Training shape:", X_train_final.shape)
    print("Testing shape:", X_test_final.shape)


if __name__ == "__main__":
    preprocess_data()