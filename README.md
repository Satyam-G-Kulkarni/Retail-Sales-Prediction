# Retail Sales Prediction — Rossmann Stores

A machine learning project to forecast sales across Rossmann retail stores, containerized with Docker and deployed using Kubernetes.

---

## Overview

Accurate sales forecasting is critical for retail operations — it directly impacts inventory management, staffing, and promotional planning. This project builds a robust sales prediction pipeline on the Rossmann Stores dataset (~1M records across 1,115 stores), identifying key demand drivers and delivering a production-ready forecasting model containerized for scalable deployment.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Models | Linear Regression, Decision Tree, Random Forest |
| Libraries | scikit-learn, pandas, numpy, matplotlib, seaborn |
| Web App | Flask |
| Containerization | Docker |
| Orchestration | Kubernetes |

---

## Dataset

**Rossmann Store Sales** — historical sales data from 1,115 Rossmann drug stores across Germany.

| Feature | Description |
|---|---|
| Store | Unique store identifier |
| Sales | Daily turnover (target variable) |
| Customers | Number of customers on a given day |
| Open | Whether the store was open |
| Promo | Whether a promotion was running |
| StateHoliday | State holiday indicator |
| SchoolHoliday | School holiday indicator |
| StoreType | Store category (a, b, c, d) |
| Assortment | Assortment level (basic, extra, extended) |
| CompetitionDistance | Distance to nearest competitor |

---

## Pipeline

```
Raw Data (store.csv + Rossmann Stores Data.csv)
        │
        ▼
Data Cleaning & Merging
(NaN handling, duplicate removal, outlier treatment)
        │
        ▼
Feature Engineering
(Date extraction: day, month, year)
(One-hot encoding of categorical variables)
(Standard scaling)
        │
        ▼
Model Training & Comparison
(Linear Regression | Decision Tree | Random Forest)
        │
        ▼
Flask Prediction App
        │
        ▼
Docker Container → Kubernetes Deployment
```

---

## Model Performance

Random Forest Regressor was selected as the final model based on its superior generalization performance.

| Metric | Train | Test |
|---|---|---|
| MAE | 0.208 | 0.249 |
| RMSE | 0.293 | 0.332 |
| R² | 0.914 | 0.884 |
| Adjusted R² | 0.914 | 0.884 |

The small train-test gap across all metrics confirms the model generalizes well without overfitting. An R² of **0.884 on unseen data** means the model explains 88.4% of variance in sales — strong performance for a retail forecasting problem with significant external noise.

---

## Key Findings

- **Promotions** have a measurable positive impact on daily sales
- **Store Type B**, despite being fewer in number, consistently achieves the highest average sales
- **Assortment Type B (Extra)** shows higher demand compared to basic and extended assortments
- **Competition distance** is a significant predictor — closer competitors correlate with lower sales
- **School holidays** show a slight positive effect on sales, likely due to increased foot traffic

---

## Deployment

The prediction app is containerized using Docker and configured for Kubernetes deployment.

### Run Locally with Docker

```bash
# Build the image
docker build -t retail-sales-prediction .

# Run the container
docker run -p 5000:5000 retail-sales-prediction
```

### Deploy on Kubernetes

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Run the Flask App Directly

```bash
pip install -r requirements.txt
python Sales_prediction_app.py
```

---

## Project Structure

```
Retail-Sales-Prediction/
├── Project_Retail_Sales_Predictions.ipynb  # Full EDA, modeling, evaluation
├── Sales_prediction_app.py                 # Flask prediction app
├── Dockerfile                              # Container configuration
├── deployment.yaml                         # Kubernetes deployment manifest
├── service.yaml                            # Kubernetes service manifest
├── requirements.txt                        # Python dependencies
├── store.csv                               # Store metadata
└── README.md
```

---

## Future Work

- Integrate external data sources (economic indicators, weather, local events)
- Implement time-series models (LSTM, Prophet) for sequential pattern capture
- Add model monitoring and drift detection in production
- Store-level fine-tuning for individual store accuracy improvement

---

## Author

**Satyam Kulkarni**
ML Engineer | MSc AI & ML
[LinkedIn](https://www.linkedin.com/in/satyam-kulkarni-92004215b/) • [GitHub](https://github.com/Satyam-G-Kulkarni)
