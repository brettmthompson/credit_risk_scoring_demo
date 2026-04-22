# Credit Risk Scoring Demo

This demo showcases an ensemble credit risk scoring system using MLServer on Red Hat OpenShift AI.

## Overview

Three machine learning models work together in an ensemble to assess credit risk:
- **XGBoost**: Gradient boosting (sequential trees) with complex non-linear interactions
- **LightGBM**: Gradient boosting (sequential trees) with native categorical handling
- **Scikit-learn**: Random Forest (parallel trees, bagging) for ensemble diversity

**Ensemble Strategy**: All models receive the same features but analyze them differently. Diversity comes from different algorithms (boosting vs bagging) making different predictions on the same data, which when combined, produces more robust results than any single model.

Models are trained in workbench notebooks (orchestrated via Data Science Pipelines), stored in SeaweedFS S3-compatible storage, and served via MLServer single-model serving platform.

## Security Notice

**This is a demo configuration.** For simplicity, this demo:
- Disables SSL certificate verification
- Disables token authentication on model endpoints

## Architecture

```
Workbench (Training)           SeaweedFS S3 Storage          MLServer (Serving)
┌─────────────────┐           ┌──────────────┐              ┌─────────────────┐
│ XGBoost Model   │──upload──>│ xgboost/     │<──load───────│ XGBoost         │
│ Training        │           │ model.ubj    │              │ InferenceService│
└─────────────────┘           └──────────────┘              └─────────────────┘

┌─────────────────┐           ┌──────────────┐              ┌─────────────────┐
│ LightGBM Model  │──upload──>│ lightgbm/    │<──load───────│ LightGBM        │
│ Training        │           │ model.bst    │              │ InferenceService│
└─────────────────┘           └──────────────┘              └─────────────────┘

┌─────────────────┐           ┌──────────────┐              ┌─────────────────┐
│ Sklearn Model   │──upload──>│ sklearn/     │<──load───────│ Sklearn         │
│ Training        │           │ model.joblib │              │ InferenceService│
└─────────────────┘           └──────────────┘              └─────────────────┘
```

## Dataset

**Kaggle Credit Risk Dataset**
- ~32,000 credit applications
- Features: person_age, person_income, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, cb_person_cred_hist_length, etc.
- Target: loan_status (default / non-default)

**Features**: All models use all 11 features from the dataset:
- person_age, person_income, person_home_ownership, person_emp_length
- loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income
- cb_person_default_on_file, cb_person_cred_hist_length

**Why this works:** Different algorithms extract different patterns from the same data. XGBoost builds sequential trees correcting errors, LightGBM efficiently handles categoricals with leaf-wise growth, and Random Forest uses parallel trees with bootstrap sampling for diversity. Combining their predictions leverages each algorithm's strengths.

## Deployment Workflow

Follow these steps to deploy the credit risk scoring demo:

### Prerequisites
- Access to an OpenShift cluster with Red Hat OpenShift AI (version 3.4 or higher) installed

### Step 1: Create Data Science Project
Navigate to the RHOAI UI and create a new data science project where all components will be deployed, e.g. `mlserver-demo`

### Step 2: Deploy SeaweedFS
Deploy SeaweedFS S3-compatible storage within your data science project. This single SeaweedFS instance will store both model files and pipeline artifacts in separate buckets.

```bash
oc apply -f resources/seaweedfs.yaml -n mlserver-demo
```

This creates:
- PersistentVolumeClaim for data storage (10Gi)
- Secret with credentials
- SeaweedFS deployment with S3 API on port 8333
- Internal SeaweedFS service

### Step 3: Create Data Connections
Create two S3 data connections in your data science project, both pointing to the same SeaweedFS service:

**Data Connection 1 - Models Bucket**
- Name: `seaweedfs-models`
- Endpoint: `http://seaweedfs-service.mlserver-demo.svc.cluster.local:8333`
- Bucket: `models`
- Access Key
- Secret Key
- Region: `us-east-1`

**Data Connection 2 - Pipelines Bucket**
- Name: `seaweedfs-pipelines`
- Endpoint: `http://seaweedfs-service.mlserver-demo.svc.cluster.local:8333`
- Bucket: `pipelines`
- Access Key
- Secret Key
- Region: `us-east-1`

### Step 4: Create Pipeline Server
Create a Data Science Pipeline server in your project:
- Attach the `seaweedfs-pipelines` data connection for artifact storage

### Step 5: Create Workbench
Create an OpenShift AI workbench in your project:
- Image: Jupyter | DataScience | CPU | Python3.12
- Attach the `seaweedfs-models` data connection for model storage

### Step 6: Clone Repository
Open your workbench and clone this repository

### Step 7: Run Training Pipeline
Submit and run the Data Science Pipeline:
1. Navigate to the `pipelines/` directory in your workbench
2. Open the `mlserver-demo.pipeline` file
3. Click "Run Pipeline" to submit it to the pipeline server
4. Monitor pipeline execution in the Pipelines UI

The pipeline orchestrates:
- Data preparation (00_data_preparation.ipynb)
- Model training in parallel:
  - XGBoost training (01_train_xgboost.ipynb)
  - LightGBM training (02_train_lightgbm.ipynb)
  - Sklearn training (03_train_sklearn.ipynb)

Models are automatically uploaded to SeaweedFS at:
- `s3://models/xgboost/model.ubj`
- `s3://models/lightgbm/model.bst`
- `s3://models/sklearn/model.joblib`

**Verify Upload (Optional):** View uploaded models in the SeaweedFS Filer UI:
```bash
oc port-forward -n mlserver-demo svc/seaweedfs-service 8888:8888
```
Then browse to `http://localhost:8888` and navigate to the `models/` bucket.

### Step 8: Deploy Models
Create three InferenceService resources, one for each model:

**XGBoost InferenceService:**
- Data connection: `seaweedfs-models`
- Path: `xgboost`
- Model Type: `Predictive`
- Model framework: `xgboost - 2`
- Runtime: `MLServer ServingRuntime for Kserve`
- Make the deployment available through an external route
- Do not require token authentication

**LightGBM InferenceService:**
- Data connection: `seaweedfs-models`
- Path: `lightgbm`
- Model Type: `Predictive`
- Model framework: `lightgbm - 4`
- Runtime: `MLServer ServingRuntime for Kserve`
- Make the deployment available through an external route
- Do not require token authentication

**Sklearn InferenceService:**
- Data connection: `seaweedfs-models`
- Path: `sklearn`
- Model Type: `Predictive`
- Model framework: `sklearn - 1`
- Runtime: `MLServer ServingRuntime for Kserve`
- Make the deployment available through an external route
- Do not require token authentication
- Custom runtime environmental variables: `MLSERVER_MODEL_EXTRA` = `{"predict_fn": "predict_proba"}`

### Step 9: Run Ensemble Application
Run the ensemble scoring application locally to interact with all three deployed models through an interactive Streamlit UI.

See [app/README.md](app/README.md) for detailed installation, configuration, and usage instructions.

## Notebooks

Run in order:

### 1. `00_data_preparation.ipynb`
- Downloads Kaggle Credit Risk dataset
- Exploratory data analysis
- Handles missing values and encodes categorical features
- **Prepares full feature set (11 features)** for all models
- Train/test split (stratified, 80/20)
- Saves datasets to local workbench storage

### 2. `01_train_xgboost.ipynb`
- Loads full feature set (11 features)
- Trains XGBoost classifier with class imbalance handling
- Hyperparameter tuning with early stopping
- Model evaluation (training progress, ROC curve, confusion matrix, feature importance)
- Saves model as `xgboost-model.ubj`
- Uploads to SeaweedFS at `xgboost/model.ubj`

### 3. `02_train_lightgbm.ipynb`
- Loads full feature set (11 features)
- Trains LightGBM classifier with native categorical support and class imbalance handling
- Specifies categorical features for efficient native handling
- Hyperparameter tuning with early stopping
- Model evaluation (training progress, ROC curve, confusion matrix, feature importance)
- Saves model as `lightgbm-model.bst`
- Uploads to SeaweedFS at `lightgbm/model.bst`

### 4. `03_train_sklearn.ipynb`
- Loads full feature set (11 features)
- Trains Random Forest classifier (300 trees, max_depth=12) with balanced class weights
- Provides ensemble diversity (bagging vs boosting from XGBoost/LightGBM)
- Model evaluation (ROC curve, confusion matrix, feature importance)
- Saves model as `sklearn-model.joblib`
- Uploads to SeaweedFS at `sklearn/model.joblib`

## Environment Variables

When you attach data connections to your workbench and pipeline server, these environment variables are automatically injected:

- `AWS_S3_ENDPOINT` = SeaweedFS service endpoint 
- `AWS_S3_BUCKET` = bucket name for storage
- `AWS_ACCESS_KEY_ID` = access key
- `AWS_SECRET_ACCESS_KEY` = secret key
- `AWS_DEFAULT_REGION` = AWS region

The training notebooks use these environment variables to connect to SeaweedFS and upload trained models. The pipeline server uses them to store pipeline artifacts.

**Note:** The Kaggle Credit Risk Dataset used in this demo is public and does not require authentication credentials.

## Python Packages

Packages are installed automatically by each notebook. Required packages:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm boto3 kaggle matplotlib seaborn joblib
```

## Business Value

This demo showcases:
- **Multi-model ensemble**: Better predictions than single models
- **Scalable serving**: MLServer handles production inference
- **Lightweight S3 storage**: SeaweedFS for efficient model storage
- **Industry-standard tooling**: boto3 for S3 interactions
- **Automated MLOps**: Workbench → Storage → Serving workflow
- **Real-time decisions**: Sub-second credit risk scoring
- **Model versioning**: Easy updates via S3-compatible storage
