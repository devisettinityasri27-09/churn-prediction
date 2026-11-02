
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================================
# MLFLOW CONFIGURATION
# ==========================================================

def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


# ==========================================================
# RUN PROJECT SCRIPTS
# ==========================================================

def run_script(script_path):
    script = PROJECT_ROOT / script_path

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT
    )

    return result.returncode


# ==========================================================
# RESTORE BACKUP FILE
# ==========================================================

def restore_file(backup_path, original_path):

    if os.path.exists(original_path):
        os.remove(original_path)

    os.replace(backup_path, original_path)


# ==========================================================
# SCENARIO 1: MISSING DATA
# ==========================================================

def test_missing_data(config):

    print("\n[LAB 9] Scenario 1: Missing data")

    data_path = PROJECT_ROOT / config["data"]["raw_path"]

    backup_path = Path(
        str(data_path) + ".lab9_backup"
    )

    if not data_path.exists():

        return {
            "scenario": "missing_data",
            "expected": "HALT",
            "actual": "NOT_TESTED",
            "passed": False
        }

    os.replace(data_path, backup_path)

    try:

        exit_code = run_script(
            "src/mlops_pipeline/02_validate_data.py"
        )

        passed = exit_code != 0

        return {
            "scenario": "missing_data",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed
        }

    finally:

        restore_file(
            backup_path,
            data_path
        )


# ==========================================================
# SCENARIO 2: INVALID DATA TYPE
# ==========================================================

def test_invalid_datatype(config):

    print("\n[LAB 9] Scenario 2: Invalid datatype")

    data_path = PROJECT_ROOT / config["data"]["raw_path"]

    backup_path = Path(
        str(data_path) + ".lab9_backup"
    )

    shutil.copy2(
        data_path,
        backup_path
    )

    try:

        df = pd.read_csv(data_path)

        df["MonthlyCharges"] = (
            df["MonthlyCharges"].astype(object)
        )

        df.loc[0, "MonthlyCharges"] = "INVALID_NUMBER"

        df.to_csv(
            data_path,
            index=False
        )

        exit_code = run_script(
            "src/mlops_pipeline/02_validate_data.py"
        )

        passed = exit_code != 0

        return {
            "scenario": "invalid_datatype",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed
        }

    finally:

        restore_file(
            backup_path,
            data_path
        )


# ==========================================================
# SCENARIO 3: SCHEMA MISMATCH
# ==========================================================

def test_schema_mismatch(config):

    print("\n[LAB 9] Scenario 3: Schema mismatch")

    data_path = PROJECT_ROOT / config["data"]["raw_path"]

    backup_path = Path(
        str(data_path) + ".lab9_backup"
    )

    shutil.copy2(
        data_path,
        backup_path
    )

    try:

        df = pd.read_csv(data_path)

        df = df.drop(
            columns=["Contract"]
        )

        df.to_csv(
            data_path,
            index=False
        )

        exit_code = run_script(
            "src/mlops_pipeline/02_validate_data.py"
        )

        passed = exit_code != 0

        return {
            "scenario": "schema_mismatch",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed
        }

    finally:

        restore_file(
            backup_path,
            data_path
        )


# ==========================================================
# SCENARIO 4: PREPROCESSING INCONSISTENCY
# ==========================================================

def test_preprocessing_inconsistency(config):

    print(
        "\n[LAB 9] Scenario 4: "
        "Preprocessing inconsistency"
    )

    processed_dir = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
    )

    path = processed_dir / "X_train_final.npy"

    backup_path = Path(
        str(path) + ".lab9_backup"
    )

    if not path.exists():

        print(
            "[ERROR] Processed training output missing. "
            "Run the normal pipeline first."
        )

        return {
            "scenario": "preprocessing_inconsistency",
            "expected": "HALT",
            "actual": "NOT_TESTED",
            "passed": False
        }

    shutil.copy2(
        path,
        backup_path
    )

    try:

        X_train = np.load(path)

        X_train[0, 0] = np.nan

        np.save(
            path,
            X_train
        )

        exit_code = run_script(
            "src/mlops_pipeline/04_validate_outputs.py"
        )

        passed = exit_code != 0

        return {
            "scenario": "preprocessing_inconsistency",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed
        }

    finally:

        restore_file(
            backup_path,
            path
        )


# ==========================================================
# SCENARIO 5: MISSING MODEL + RECOVERY
# ==========================================================

def test_missing_model_with_recovery(config):

    print(
        "\n[LAB 9] Scenario 5: "
        "Missing model artifact + recovery"
    )

    model_path = (
        PROJECT_ROOT
        / config["paths"]["model"]
    )

    model_name = config["registry"]["model_name"]

    if not model_path.exists():

        print(
            "[ERROR] Local model is missing before test."
        )

        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "NOT_TESTED",
            "passed": False
        }

    backup_path = Path(
        str(model_path) + ".lab9_backup"
    )

    shutil.copy2(
        model_path,
        backup_path
    )

    os.remove(model_path)

    try:

        # Step 1: Check whether missing model is detected.

        failure_exit_code = run_script(
            "src/mlops_pipeline/06_model_evaluation.py"
        )

        failure_detected = (
            failure_exit_code != 0
        )

        # Step 2: Attempt recovery from MLflow.

        configure_mlflow(config)

        production_uri = (
            f"models:/{model_name}/Production"
        )

        print(
            f"[INFO] Attempting recovery from "
            f"{production_uri}..."
        )

        recovered = False

        try:

            production_model = (
                mlflow.sklearn.load_model(
                    production_uri
                )
            )

            model_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            joblib.dump(
                production_model,
                model_path
            )

            # Step 3: Evaluate after recovery.

            recovery_exit_code = run_script(
                "src/mlops_pipeline/06_model_evaluation.py"
            )

            recovered = (
                recovery_exit_code == 0
            )

        except Exception as error:

            print(
                f"[WARNING] MLflow recovery unavailable: "
                f"{error}"
            )

            recovered = False

        # The test passes if the missing model
        # was correctly detected.
        # Recovery is reported separately because
        # GitHub Actions may not have a remote registry.

        return {
            "scenario": "missing_model_artifact",
            "expected": "DETECT_AND_RECOVER",
            "actual": (
                "DETECTED_AND_RECOVERED"
                if failure_detected and recovered
                else (
                    "DETECTED_RECOVERY_UNAVAILABLE"
                    if failure_detected
                    else "FAILURE_NOT_DETECTED"
                )
            ),
            "failure_detected": failure_detected,
            "recovery_succeeded": recovered,
            "passed": failure_detected
        }

    finally:

        # Always restore the original local model.

        if model_path.exists():
            os.remove(model_path)

        restore_file(
            backup_path,
            model_path
        )


# ==========================================================
# RUN ALL LAB 9 FAILURE TESTS
# ==========================================================

def run_failure_tests():

    print("=" * 65)

    print(
        "[INFO] Lab 9: Failure Handling "
        "and Recovery Tests"
    )

    print("=" * 65)

    # Load configuration.

    config_path = (
        PROJECT_ROOT / "config.yaml"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    repetitions = int(
        config["lab9"]["repetitions"]
    )

    all_runs = []

    # Lab 9 test scenarios.

    tests = [
        test_missing_data,
        test_invalid_datatype,
        test_schema_mismatch,
        test_preprocessing_inconsistency,
        test_missing_model_with_recovery
    ]

    # Repeat testing as configured.

    for repetition in range(
        1,
        repetitions + 1
    ):

        print("\n" + "=" * 65)

        print(
            f"[INFO] Lab 9 Repetition "
            f"{repetition}/{repetitions}"
        )

        print("=" * 65)

        repetition_results = []

        
        for test in tests:

            result = test(config)

            repetition_results.append(result)

            print(
                f"[RESULT] {result['scenario']}: "
                f"{result['actual']}"
            )


        all_runs.append(
            {
                "repetition": repetition,
                "results": repetition_results,
                "passed": all(
                    item["passed"]
                    for item in repetition_results
                )
            }
        )

    # Generate Lab 9 report.

    report = {

        "stage": (
            "Lab 9 - Failure Handling "
            "and Recovery"
        ),

        "repetitions": repetitions,

        "runs": all_runs,

        "overall_status": (
            "PASSED"
            if all(
                run["passed"]
                for run in all_runs
            )
            else "FAILED"
        )
    }

    reports_dir = (
        PROJECT_ROOT / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        PROJECT_ROOT
        / config["paths"]["failure_report"]
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # Display final status.

    print("\n" + "=" * 65)

    print("[INFO] LAB 9 FINAL RESULTS")

    print("=" * 65)

    print(
        f"[INFO] Total repetitions: "
        f"{repetitions}"
    )

    print(
        f"[INFO] Overall status: "
        f"{report['overall_status']}"
    )

    print(
        f"[INFO] Report saved to: "
        f"{report_path}"
    )

    if report["overall_status"] == "PASSED":

        print(
            "\n[SUCCESS] Lab 9 failure tests completed!"
        )

        return True

    print(
        "\n[WARNING] Some failure scenarios "
        "need attention."
    )

    return False


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    passed = run_failure_tests()

    sys.exit(
        0 if passed else 1
    )
