# Credit Risk Ensemble Prediction App

Interactive Streamlit application for predicting loan default risk using an ensemble of three machine learning models deployed via MLServer.

**Note:** This is Step 9 of the deployment workflow. Complete Steps 1-8 in the [main README](../README.md) before running this application.

**Security Notice:** For simplicity, this demo application disables SSL certificate verification and does not require authentication tokens. This configuration is for demonstration purposes only.

## Features

- **Interactive UI**: Sliders and inputs for all 11 credit risk features
- **Multi-Model Ensemble**: Parallel calls to XGBoost, LightGBM, and Scikit-learn Random Forest models
- **Configurable Threshold**: Adjustable decision threshold via UI slider (default 0.5)
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Visual Results**: Color-coded risk indicators and model agreement analysis
- **Editable Configuration**: All model endpoints and names configurable in sidebar

## Prerequisites

1. **Trained models deployed** on OpenShift AI as InferenceServices (from Step 8 of deployment workflow):
   - XGBoost InferenceService with external route enabled
   - LightGBM InferenceService with external route enabled
   - Sklearn InferenceService with external route enabled

2. **Model endpoints accessible** from where you run the app (routes must be externally accessible)

3. **Python 3.10+** installed

## Installation

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the app
uv run streamlit run ensemble_app.py
```

### Using pip

```bash
pip install streamlit requests

# Run the app
streamlit run ensemble_app.py
```

## Configuration

### Configuration

All model endpoints and names are configured through the Streamlit UI sidebar - no code changes needed!

**UI Configuration:**
1. **Start the app**: `uv run streamlit run ensemble_app.py`
2. **Open sidebar**: Click the `>` icon in the top-left
3. **Configure endpoints**: Enter the external route URLs from your deployed InferenceServices (Step 8)
4. **Configure model names**: Enter the names of your InferenceServices from Step 8

**Environment Variables (Optional - for convenience):**

To avoid re-entering URLs every time you restart the app, set environment variables to pre-populate the UI fields:

```bash
export XGBOOST_ENDPOINT="<your-xgboost-inferenceservice-route>"
export LIGHTGBM_ENDPOINT="<your-lightgbm-inferenceservice-route>"
export SKLEARN_ENDPOINT="<your-sklearn-inferenceservice-route>"

export XGBOOST_MODEL_NAME="<your-xgboost-model-name>"
export LIGHTGBM_MODEL_NAME="<your-lightgbm-model-name>"
export SKLEARN_MODEL_NAME="<your-sklearn-model-name>"

uv run streamlit run ensemble_app.py
```

**Note:** 
- Endpoints can use HTTP or HTTPS (SSL verification is disabled for demo purposes)
- You can still edit values in the UI even when environment variables are set
- No authentication tokens required (demo configuration from Step 8)

## Usage

1. **Start the app**: `uv run streamlit run ensemble_app.py`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Enter applicant details**:
   - Personal: Age, income, home ownership, employment length
   - Loan: Purpose, grade, amount, interest rate
   - Credit history: Previous defaults, credit history length
4. **Click "Predict Default Risk"**
5. **View results**:
   - Individual model predictions
   - Ensemble probability
   - Approve/Deny recommendation

## How It Works

1. **Feature Encoding**: Categorical features are encoded using the same mapping as training
2. **Parallel API Calls**: Sends inference requests to all three MLServer endpoints simultaneously using V2 protocol
3. **Output Handling**: Extracts positive class probability from different output formats (sklearn returns [P(class0), P(class1)], others return single probability)
4. **Ensemble Logic**: Averages the probability predictions from all models
5. **Decision**: Compares ensemble probability against configurable threshold (default 0.5, adjustable via slider)

## Troubleshooting

**Connection errors?**
- Verify model endpoints are accessible
- Check InferenceServices are running: `oc get inferenceservices`
- Ensure network connectivity to cluster

**Wrong predictions?**
- Verify feature encoding matches training data
- Check model versions match trained models
- Ensure sklearn model is configured with `MLSERVER_MODEL_EXTRA='{"predict_fn": "predict_proba"}'` to return probabilities
- Review inference request format

**Models not found?**
- Confirm models are deployed at the specified endpoints
- Check model names match your InferenceService names from Step 8
- Verify external routes are enabled on all three InferenceServices

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│  (User Input)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Encode  │
    │Features │
    └────┬────┘
         │
    ┌────┴──────────────────┐
    │                       │
┌───▼────┐  ┌────▼────┐  ┌─▼──────┐
│XGBoost │  │LightGBM │  │Sklearn │
│MLServer│  │MLServer │  │MLServer│
└───┬────┘  └────┬────┘  └─┬──────┘
    │            │          │
    └────────┬───┴──────────┘
         ┌───▼────┐
         │Ensemble│
         │Average │
         └───┬────┘
         ┌───▼────┐
         │Result  │
         │Display │
         └────────┘
```
