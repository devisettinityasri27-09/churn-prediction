# Lab 11 — Containerization and Kubernetes Deployment Report

## 1. Objective

The objective of Lab 11 was to containerize the Churn Prediction ML application, validate its portability and prediction consistency, deploy the containerized API using Kubernetes, demonstrate scaling, simulate deployment failures, and validate recovery.

---

## 2. Deployment Architecture

The deployment architecture follows this flow:

```text
MLflow
   ↓
Production Model (Version 4)
   ↓
Matching Model + Preprocessor
   ↓
Deployment Bundle
   ↓
Docker Image (v2)
   ↓
Kubernetes Deployment
   ↓
Kubernetes Pods
   ↓
Kubernetes Service
   ↓
FastAPI API
```

MLflow is used as the model lifecycle management and model version tracking system.

The deployment bundle contains the model and preprocessing artifacts associated with the Production model.

---

## 3. MLflow Production Model

**Model Name:**  
Telco_Churn_Production_Model

**Production Version:**  
4

**Stage:**  
Production

The deployment preparation process retrieved the Production model from MLflow and extracted the corresponding model and preprocessor artifacts.

The model and preprocessing artifacts were packaged together into the deployment bundle to ensure consistent inference during containerized deployment.

MLflow served as the model lifecycle management and model version tracking system.

---

## 4. Deployment Bundle

The deployment bundle contains:

```text
deployment/
├── model.pkl
├── preprocessor.pkl
└── model_metadata.json
```

The metadata records the MLflow model name, version, stage, Run ID, model ID, and source information.

This allows the deployed model to retain its provenance and traceability.

---

## 5. Docker Containerization

Docker was used to package the FastAPI application together with its runtime dependencies and verified ML artifacts.

**Docker image:**

`churn-prediction-api:v2`

The API container contains:

- FastAPI
- Uvicorn
- Pydantic
- NumPy
- Pandas
- scikit-learn
- joblib
- API application code
- Deployment bundle
- Configuration file

MLflow is not required during API runtime because the deployment bundle is included directly in the Docker image.

---

## 6. Docker Image Creation

The Docker image was built using:

```powershell
docker build -t churn-prediction-api:v2 .
```

The image was successfully created.

The containerized FastAPI application was executed using Docker and exposed through a host port.

The application successfully started inside the container.

---

## 7. Containerized API Validation

The containerized API was tested using the following endpoints:

- `GET /`
- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict/batch`

The health endpoint returned:

```text
status = healthy
```

The metadata endpoint confirmed:

```text
service = Churn Prediction API
model_name = Telco_Churn_Production_Model
model_loaded = True
preprocessor_loaded = True
```

The prediction and batch prediction endpoints were also tested successfully.

---

## 8. Local vs Container Prediction Consistency

The same customer input was sent to the local API and the containerized API.

The model and preprocessor artifacts were verified to be identical using SHA-256 checksums.

The installed scikit-learn and pandas versions were also checked.

However, the observed prediction results were different:

| Metric | Local API | Container API |
|---|---|---|
| Prediction | 1 | 0 |
| Churn probability | 0.5171495 | 0.49339275 |

**Result:** Prediction consistency was not achieved for the tested input.

This discrepancy was identified for further investigation. Therefore, complete local-versus-container prediction consistency cannot be claimed based on this test.

---

## 9. Deployment Failure Simulation

A missing model artifact failure was intentionally simulated during containerization testing.

The model-loading process was tested with the required model artifact unavailable.

During application startup, the ModelLoader attempted to load:

```text
/app/deployment/model.pkl
```

The application encountered:

```text
FileNotFoundError: Model not found
```

This demonstrated a deployment failure caused by a missing model artifact.

---

## 10. Failure Debugging

The failed container was inspected using Docker commands.

The container failure was traced to the missing model artifact required during application startup.

**Root cause:**

The required model file was unavailable in the deployment bundle inside the container.

This prevented the FastAPI application from successfully loading the model and starting inference.

---

## 11. Deployment Recovery

The failed container was removed and recreated using the previously built Docker image.

The recovery process used the known-good container image:

```text
churn-prediction-api:v2
```

After recreating the container, the application successfully loaded:

- Model
- Preprocessor
- Model metadata

FastAPI successfully started after recovery.

The health endpoint was subsequently validated.

---

## 12. Kubernetes Deployment

Kubernetes was enabled using Docker Desktop.

**Kubernetes version:**  
v1.36.1

**Deployment name:**  
churn-api

**Docker image:**  
churn-prediction-api:v2

**Initial replicas:**  
1

The Kubernetes Deployment was created using:

```text
k8s/deployment.yaml
```

The deployment successfully created a running Pod.

The Pod status was verified using:

```powershell
kubectl get pods
```

---

## 13. Kubernetes Service

A Kubernetes Service was created using:

```text
k8s/service.yaml
```

**Service name:**  
churn-api-service

**Service type:**  
NodePort

**Service port:**  
8000

**Target port:**  
8000

**NodePort:**  
32266

**Cluster IP:**  
10.96.214.17

The Service selected Pods using the label:

```text
app=churn-api
```

The Service successfully connected to the FastAPI Pods.

The application was accessed and validated using Kubernetes port forwarding.

---

## 14. Kubernetes Scaling

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

Two Pods were running simultaneously.

This demonstrated Kubernetes horizontal scaling.

---

## 15. Kubernetes Service Interruption

One running Pod was intentionally deleted to simulate a service interruption.

The command used was:

```powershell
kubectl delete pod <pod-name>
```

Kubernetes detected that the actual number of Pods was below the desired replica count.

A replacement Pod was automatically created by the Deployment controller.

The final state returned to two running Pods.

This demonstrated Kubernetes self-healing behavior.

---

## 16. Kubernetes Recovery Validation

After the Pod deletion and automatic recovery, the Pods were checked using:

```powershell
kubectl get pods
```

Two Pods were running successfully.

The Kubernetes Service was accessed using port forwarding:

```powershell
kubectl port-forward service/churn-api-service 8001:8000
```

The health endpoint was validated using:

```powershell
Invoke-RestMethod http://localhost:8001/health
```

The response was:

```text
status
------
healthy
```

Therefore, Kubernetes successfully recovered the deleted Pod, and the application remained operational.

---

## 17. Lab 11 Results

The following Lab 11 requirements were demonstrated:

- ML deployment containerization
- Docker image creation
- Dependency packaging
- FastAPI container execution
- Containerized API validation
- Local-versus-container prediction comparison
- Deployment failure simulation
- Failure diagnosis
- Deployment recovery
- Kubernetes deployment
- Kubernetes Service exposure
- Kubernetes horizontal scaling
- Kubernetes Pod interruption
- Kubernetes automatic recovery
- Post-recovery API validation

---

## 18. Final Architecture

```text
MLflow
   ↓
Production Model (Version 4)
   ↓
Deployment Bundle
   ↓
Docker Image (v2)
   ↓
Kubernetes Deployment
   ↓
2 FastAPI Pods
   ↓
Kubernetes NodePort Service
   ↓
Client
```

The system demonstrates an end-to-end ML deployment workflow from model lifecycle management to containerized and Kubernetes-managed inference.

---

## 19. Next Step

The remaining Lab 11 activities are:

**11.19 — Deployment and Debugging Report**

**11.20 — Final Lab 11 Validation**

After completing Lab 11, Lab 12 will focus on:

- Monitoring
- Inference logging
- Prediction distributions
- Latency and throughput
- Failure monitoring
- Data drift
- Prediction drift
- Drift analysis
- Retraining
- Model comparison
- Updated model validation
- Deployment consistency
