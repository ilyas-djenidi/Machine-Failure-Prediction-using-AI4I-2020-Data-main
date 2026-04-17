# Predictive Maintenance ML System 🤖⚙️

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Production-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)

> Professional predictive maintenance system that predicts machine failures 1-7 days in advance using multi-dataset ML models with SHAP interpretability and automated PDF reporting.

---

## 🚀 Key Features

- ✅ **Multi-Dataset Integration**: Combines 4 industrial datasets (UCI AI4I, NASA CMAPS, Azure, Kaggle)
- ✅ **Advanced ML Models**: Random Forest, XGBoost, LightGBM, Isolation Forest
- ✅ **Feature Engineering**: 100+ features from rolling statistics, lag, trends, degradation indicators
- ✅ **SHAP Interpretability**: Root cause analysis for every prediction
- ✅ **Professional PDF Reports**: Auto-generated reports with charts and maintenance recommendations
- ✅ **Risk-Based Classification**: CRITICAL/HIGH/MEDIUM/LOW failure risk levels
- ✅ **Production-Ready**: Modular code, configuration-driven, comprehensive logging

---

## 📁 Project Structure

```
predictive-maintenance/
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Preprocessed data
│   └── datasets_info.md        # Dataset documentation
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Dataset loading & caching
│   │   └── preprocessing.py    # Data cleaning pipeline
│   ├── features/
│   │   └── feature_engineering.py  # Feature creation
│   ├── models/
│   │   ├── train.py            # Model training
│   │   ├── evaluate.py         # Evaluation & metrics
│   │   └── predict.py          # Prediction system
│   ├── reports/
│   │   └── generate_report.py  # PDF report generator
│   └── app.py                  # Streamlit dashboard
├── models/                     # Saved trained models
├── reports/                    # Generated PDF reports
├── notebooks/                  # Jupyter notebooks
├── config.yaml                 # System configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Machine-Failure-Prediction-using-AI4I-2020-Data-main
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download datasets**:
```bash
python download_datasets.py
```

> **Note**: For Kaggle datasets, you'll need to configure your Kaggle API credentials. See [Kaggle API docs](https://www.kaggle.com/docs/api).

---

## 📊 Datasets

| Dataset | Status | Samples | Features | Use Case |
|---------|--------|---------|----------|----------|
| **UCI AI4I 2020** | ✅ | ~10,000 | 5 | Primary training dataset |
| **NASA CMAPS** | ⚠️ | Varies | 21 sensors | Time-series/LSTM training |
| **Azure Maintenance** | ⚠️ | Multi-table | Various | Complex industrial scenarios |
| **Kaggle Machine Failure** | ⚠️ | Varies | Various | Additional validation |

---

## 🎯 Usage

### Quick Start: Training Pipeline

```python
# 1. Load data
from src.data.data_loader import DatasetLoader
loader = DatasetLoader()
X, y = loader.load_uci_ai4i()

# 2. Preprocess
from src.data.preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
df = pd.concat([X, y], axis=1)
splits = preprocessor.full_pipeline(df, target_col='Machine failure')

# 3. Feature engineering
from src.features.feature_engineering import FeatureEngineer
engineer = FeatureEngineer()
X_train = engineer.full_pipeline(splits['X_train'])

# 4. Train models
from src.models.train import ModelTrainer
trainer = ModelTrainer()
models = trainer.train_all_baselines(X_train, splits['y_train'])
trainer.save_all_models(models)

# 5. Evaluate
from src.models.evaluate import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models, X_test, splits['y_test'])
```

### Making Predictions

```python
from src.models.predict import FailurePredictor

predictor = FailurePredictor(model_path="models/xgboost_latest.pkl")
result = predictor.predict_comprehensive(
    sensor_data=new_sensor_readings,
    feature_names=feature_list,
    machine_id="M123"
)

print(f"Failure Probability: {result['failure_probability']}%")
print(f"Risk Level: {result['risk_level']}")
print(f"Days to Failure: {result['estimated_days_to_failure']}")
```

### Generating Reports

```python
from src.reports.generate_report import ReportGenerator

generator = ReportGenerator()
pdf_path = generator.generate_failure_report(
    machine_id="M123",
    prediction_result=result,
    sensor_data=sensor_df
)
print(f"Report saved to: {pdf_path}")
```

### Running the Streamlit Dashboard

```bash
streamlit run src/app.py
```

---

## ⚙️ Configuration

All system parameters are in `config.yaml`:

- **Data preprocessing**: missing value strategies, outlier detection, normalization
- **Feature engineering**: rolling windows, lag periods, interaction degree
- **Models**: hyperparameters for RF, XGBoost, LightGBM, Isolation Forest
- **Evaluation**: performance thresholds (precision ≥80%, recall ≥75%, F1 ≥77%)
- **Reports**: risk thresholds, styling, output paths

**No code changes needed for tuning!**

---

## 📈 Model Performance

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Precision | ≥ 80% | Configurable |
| Recall | ≥ 75% | Configurable |
| F1 Score | ≥ 77% | Configurable |
| ROC-AUC | ≥ 0.90 | Configurable |

### Available Models

1. **Logistic Regression** - Fast baseline
2. **Random Forest** - Robust ensemble  
3. **XGBoost** - High-performance boosting
4. **LightGBM** - Fast gradient boosting
5. **Isolation Forest** - Anomaly detection

---

## 📄 Reports

Generated PDFs include:

1. **Executive Summary**
   - Machine ID, timestamp, failure probability
   - Risk level (color-coded)
   - Confidence score, time to failure

2. **Risk Gauge** - Visual probability meter

3. **Root Cause Analysis**
   - Top 5 contributing sensors
   - SHAP importance scores

4. **Maintenance Recommendations**
   - Risk-specific action items
   - Urgency and procedures

5. **Technical Analysis**
   - Sensor trend charts  
   - Current sensor values

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🚀 Deployment

### Docker (Optional)

```bash
# Build image
docker build -t predictive-maintenance .

# Run container
docker run -p 8000:8000 predictive-maintenance
```

### API Deployment (Optional)

Create `src/api.py` with FastAPI:

```python
from fastapi import FastAPI
from src.models.predict import FailurePredictor

app = FastAPI()
predictor = FailurePredictor(model_path="models/xgboost_latest.pkl")

@app.post("/predict")
def predict(sensor_data: dict):
    result = predictor.predict_comprehensive(...)
    return result
```

Run with:
```bash
uvicorn src.api:app --reload
```

---

## 📚 Documentation

- **[Implementation Plan](./implementation_plan.md)**: Technical design and architecture
- **[Walkthrough](./walkthrough.md)**: Complete system overview and usage guide
- **[Dataset Info](./data/datasets_info.md)**: Dataset descriptions and sources

---

## 🔮 Future Enhancements

- [ ] LSTM model for time-series prediction
- [ ] Autoencoder for anomaly detection
- [ ] Real-time streaming data ingestion
- [ ] Retraining pipeline with model drift detection
- [ ] Multi-machine dashboard
- [ ] Alert system integration

---

## 📦 Technologies Used

| Category | Technologies |
|----------|-------------|
| **ML** | Scikit-learn, XGBoost, LightGBM, SHAP |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Reporting** | ReportLab |
| **Web** | Streamlit, FastAPI (optional) |
| **Config** | PyYAML |

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 📧 Support

For questions or issues, please open an issue on GitHub.

---

## ⭐ Acknowledgments

- UCI Machine Learning Repository for AI4I 2020 dataset
- NASA for CMAPS turbofan degradation data
- Microsoft Azure AI Gallery
- Kaggle community datasets

---

**Built with ❤️ for production-grade predictive maintenance**