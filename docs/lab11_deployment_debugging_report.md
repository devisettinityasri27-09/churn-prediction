# Lab 11.19 — Deployment and Debugging Report

## 1. Objective

The objective of this activity was to simulate deployment failures, identify their root causes, debug the failures using container logs and status information, recover the deployment, and validate Kubernetes self-healing behavior.

---

## 2. Deployment Environment

### Docker

Docker was used to package and execute the Churn Prediction API.

**Docker image:**  
`churn-prediction-api:v2`

**Application:**  
FastAPI

**Application port:**  
8000

### Kubernetes

Kubernetes was provided through Docker Desktop.

**Kubernetes version:**  
v1.36.1

**Node:**  
desktop-control-plane

**Node status:**  
Ready

---

## 3. Failure Scenario 1 — Missing Model Artifact

### 3.1 Failure Injection

A missing model artifact failure was simulated during deployment testing.

The application was tested with the required model artifact unavailable at its expected path.

Expected model path:

```text
/app/deployment/model.pkl
```

### 3.2 Failure Observed

During application startup, the ModelLoader attempted to load the required model artifact.

The application encountered:

```text
FileNotFoundError: Model not found
```

The FastAPI application failed during startup because the required model artifact was unavailable.

---

## 4. Failure Detection

The container status was inspected using Docker commands.

The container failure was identified through its status and application logs.

The application could not initialize successfully because the model-loading process failed.

The failure was confirmed through the error message indicating that the required model file was missing.

---

## 5. Root Cause Analysis

**Failure:** FastAPI application failed during model initialization.

**Expected:**

```text
/app/deployment/model.pkl
```

**Actual:** The required model artifact was unavailable at the expected path.

**Root cause:** Missing deployment model artifact.

ModelLoader checks for the required model file before loading it. Because the file was unavailable, it raised a FileNotFoundError and FastAPI startup failed.

---

## 6. Log-Based Debugging

Docker logs were inspected to identify the application startup error.

The important error was:

```text
FileNotFoundError: Model not found:
/app/deployment/model.pkl
```

The failure flow was:

```text
FastAPI lifespan
       ↓
ModelLoader initialization
       ↓
Deployment bundle loading
       ↓
Missing model.pkl
       ↓
FileNotFoundError
       ↓
Application startup failed
```

This allowed the failure to be traced from the application startup process to the missing model artifact.

---

## 7. Deployment Recovery

The failed container was removed and recreated using the known-good Docker image.

Docker image used:

```text
churn-prediction-api:v2
```

The original image was not modified during failure simulation.

The container was recreated from the existing image, allowing the application to load the deployment bundle again.

---

## 8. Recovery Result

After recreating the container, the application successfully loaded:

- Model
- Preprocessor
- Model metadata

FastAPI successfully started after recovery.

The health endpoint was validated again, confirming that the application was operational.

---

## 9. Kubernetes Deployment

The application was deployed using a Kubernetes Deployment.

**Deployment name:**

```text
churn-api
```

**Docker image:**

```text
churn-prediction-api:v2
```

The initial replica count was one.

The Deployment successfully created a running Pod.

---

## 10. Kubernetes Scaling

The Deployment was scaled from one replica to two replicas using:

```powershell
kubectl scale deployment churn-api --replicas=2
```

The resulting Deployment status was:

```text
2/2 READY
2/2 UP-TO-DATE
2/2 AVAILABLE
```

Two Pods