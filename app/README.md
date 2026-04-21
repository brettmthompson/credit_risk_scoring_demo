# Credit Risk Ensemble Prediction App

Interactive Streamlit application for predicting loan default risk using an ensemble of three machine learning models deployed via MLServer.

## Features

- **Interactive UI**: Sliders and inputs for all 11 credit risk features
- **Multi-Model Ensemble**: Parallel calls to XGBoost, LightGBM, and Scikit-learn Random Forest models
- **Configurable Threshold**: Adjustable decision threshold via UI slider (default 0.5)
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Visual Results**: Color-coded risk indicators and model agreement analysis
- **Editable Configuration**: All model endpoints and names configurable in sidebar

## Prerequisites

1. **Trained models deployed** on OpenShift AI as InferenceServices:
   - `xgboost-predictor`
   - `lightgbm-predictor`
   - `sklearn-predictor`

2. **Model endpoints accessible** from where you run the app

3. **Python 3.9+** installed

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

### UI Configuration (Recommended)

The app provides editable configuration in the sidebar:

1. **Start the app**: `uv run streamlit run ensemble_app.py`
2. **Open sidebar**: Click the `>` icon in the top-left
3. **Configure endpoints**: Edit the endpoint URLs for each model
4. **Configure model names**: Edit the model names to match your MLServer configuration

All settings are editable directly in the browser - no code changes needed!

### Environment Variables (Optional)

You can also set defaults via environment variables:

```bash
# Model endpoints
export XGBOOST_ENDPOINT="https://xgboost-predictor-namespace.apps.your-cluster.com"
export LIGHTGBM_ENDPOINT="https://lightgbm-predictor-namespace.apps.your-cluster.com"
export SKLEARN_ENDPOINT="https://sklearn-predictor-namespace.apps.your-cluster.com"

# Model names
export XGBOOST_MODEL_NAME="xgboost"
export LIGHTGBM_MODEL_NAME="lightgbm"
export SKLEARN_MODEL_NAME="sklearn"

uv run streamlit run ensemble_app.py
```

**Note:** 
- Endpoints can use HTTP or HTTPS (SSL verification is disabled for self-signed certificates)
- Model names should match the `name` field in your MLServer model configuration
- UI values override environment variables
- No authentication tokens required

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
- Check InferenceServices are running: `kubectl get inferenceservices`
- Ensure network connectivity to cluster

**Wrong predictions?**
- Verify feature encoding matches training data
- Check model versions match trained models
- Ensure sklearn model is configured with `MLSERVER_MODEL_EXTRA='{"predict_fn": "predict_proba"}'` to return probabilities
- Review inference request format

**Models not found?**
- Confirm models are deployed at the specified endpoints
- Check model names match (`xgboost`, `lightgbm`, `sklearn`)

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

## Next Steps

- Adjust decision threshold based on business costs (already configurable via slider)
- Add weighted ensemble (different model weights instead of simple average)
- Implement voting ensemble (majority vote option)
- Add confidence intervals or uncertainty quantification
- Log predictions for monitoring and model drift detection
- Add model performance metrics display
