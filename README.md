# Credit Risk Scoring Demo

This demo showcases an ensemble credit risk scoring system using MLServer on Red Hat OpenShift AI.

## Overview

Three machine learning models work together in an ensemble to assess credit risk:
- **XGBoost**: Gradient boosting (sequential trees) with complex non-linear interactions
- **LightGBM**: Gradient boosting (sequential trees) with native categorical handling
- **Scikit-learn**: Random Forest (parallel trees, bagging) for ensemble diversity

**Ensemble Strategy**: All models receive the same features but analyze them differently. Diversity comes from different algorithms (boosting vs bagging) making different predictions on the same data, which when combined, produces more robust results than any single model.

Models are trained in workbench notebooks, stored in MinIO S3, and served via MLServer single-model serving platform.

## Architecture

```
Workbench (Training)           MinIO S3 Storage              MLServer (Serving)
┌─────────────────┐           ┌──────────────┐              ┌─────────────────┐
│ XGBoost Model   │──upload──>│ xgboost-     │<──load───────│ XGBoost         │
│ Training        │           │ model.bst    │              │ InferenceService│
└─────────────────┘           └──────────────┘              └─────────────────┘

┌─────────────────┐           ┌──────────────┐              ┌─────────────────┐
│ LightGBM Model  │──upload──>│ lightgbm-    │<──load───────│ LightGBM        │
│ Training        │           │ model.bst    │              │ InferenceService│
└─────────────────┘           └──────────────┘              └─────────────────┘

┌─────────────────┐           ┌──────────────┐              ┌─────────────────┐
│ Sklearn Model   │──upload──>│ sklearn-     │<──load───────│ Sklearn         │
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

## Prerequisites

1. Red Hat OpenShift AI workbench with Jupyter
2. MinIO S3 storage configured
3. Python packages: pandas, numpy, scikit-learn, xgboost, lightgbm, minio

## S3 Storage Configuration

Configure S3-compatible storage (MinIO) for model artifacts by either:

1. **Attach an S3 Data Connection** to your workbench (recommended)
2. **Set environment variables** manually:
   - `AWS_S3_ENDPOINT`: MinIO service endpoint (e.g., `http://minio-service.mlserver-tutorial.svc.cluster.local:9000`)
   - `AWS_S3_BUCKET`: Bucket name for model storage (e.g., `models`)
   - `AWS_ACCESS_KEY_ID`: MinIO access key
   - `AWS_SECRET_ACCESS_KEY`: MinIO secret key

## Model Storage Paths

Models are uploaded to the `models` bucket organized by framework:
- `xgboost/model.ubj`
- `lightgbm/model.bst`
- `sklearn/model.joblib`

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
- Model evaluation (accuracy, AUC, confusion matrix, feature importance)
- Saves model as `xgboost-model.ubj`
- Uploads to MinIO at `xgboost/model.ubj`

### 3. `02_train_lightgbm.ipynb`
- Loads full feature set (11 features)
- Trains LightGBM classifier with native categorical support and class imbalance handling
- Specifies categorical features for efficient native handling
- Hyperparameter tuning with early stopping
- Model evaluation
- Saves model as `lightgbm-model.bst`
- Uploads to MinIO at `lightgbm/model.bst`

### 4. `03_train_sklearn.ipynb`
- Loads full feature set (11 features)
- Trains Random Forest classifier (300 trees, max_depth=12) with balanced class weights
- Provides ensemble diversity (bagging vs boosting from XGBoost/LightGBM)
- Model evaluation with feature importance analysis
- Saves model as `sklearn-model.joblib`
- Uploads to MinIO at `sklearn/model.joblib`

## Prerequisites

### 1. Install Packages

Packages will be installed automatically by each notebook, or install manually:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm minio kaggle
```

### 2. Configure Environment Variables

Set environment variables in your OpenShift AI workbench or attach an S3 Data Connection:

**MinIO S3 Configuration** (required):
- `AWS_S3_ENDPOINT` = your MinIO endpoint (e.g., `http://minio-service.mlserver-tutorial.svc.cluster.local:9000`)
- `AWS_S3_BUCKET` = bucket name for model storage (e.g., `models`)
- `AWS_ACCESS_KEY_ID` = your access key
- `AWS_SECRET_ACCESS_KEY` = your secret key

**Note:** The Kaggle Credit Risk Dataset used in this demo is public and does not require authentication credentials.

## Next Steps

After training and uploading models:

1. Create MLServer ServingRuntime resources for each model type
2. Create InferenceService resources pointing to MinIO model paths
3. Test inference endpoints
4. Build ensemble scoring logic
5. (Optional) Create inference pipeline for automated decisions

## Business Value

This demo showcases:
- **Multi-model ensemble**: Better predictions than single models
- **Scalable serving**: MLServer handles production inference
- **Automated MLOps**: Workbench → Storage → Serving workflow
- **Real-time decisions**: Sub-second credit risk scoring
- **Model versioning**: Easy updates via MinIO storage
