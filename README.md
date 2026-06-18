# Customer Churn Prediction - Telecom Industry

A complete, end-to-end machine learning project that predicts which telecom customers are likely to cancel their subscription, and explains *why*, so a retention team can act before they leave.

## Business Problem

Telecom companies lose a meaningful share of subscribers every quarter. Acquiring a new customer typically costs far more than retaining an existing one, so the retention team needs to know **which customers are at risk** and **what's driving that risk**, early enough to intervene with a targeted offer.

## Who Needs This & Why

| Stakeholder | Need |
|---|---|
| Retention Marketing Team | A ranked list of at-risk customers each cycle, not just a yes/no label, to prioritize outreach. |
| Customer Success Managers | The *reasons* behind churn risk (contract type, tenure, billing, services) so offers target the actual pain point. |
| Leadership | A measurable trade-off between catching more true churners (recall) and the cost of false alarms (precision), since outreach has a real cost per customer. |

## Dataset

[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) - 7,043 customers x 21 columns: demographics, account info (tenure, contract, billing), services subscribed (internet, streaming, tech support), and the churn label.

## Project Workflow

1. **Data Cleaning** - fixed a text-typed `TotalCharges` column with blank entries (new customers not yet billed), removed the non-predictive customer ID, checked for duplicates.
2. **Exploratory Data Analysis** - churn rate by contract type, tenure, monthly charges, and internet service type.
3. **Feature Engineering** - one-hot encoding of categorical fields, standard scaling of numeric fields, stratified 80/20 train-test split.
4. **Modeling** - Logistic Regression (interpretable baseline) and Random Forest (captures non-linear interactions), both with `class_weight='balanced'` since churners are a minority class (~27%).
5. **Evaluation** - accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and a feature-importance ranking.
6. **Business Translation** - turned model output into concrete retention recommendations.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 |
| **Random Forest** | **0.764** | **0.538** | **0.783** | **0.638** | **0.843** |

The Random Forest model was selected: same recall as Logistic Regression (catches 78% of true churners) with fewer false alarms.

**Top churn drivers:** tenure, contract length (two-year contracts retain best), total/monthly charges, fiber-optic internet, and electronic-check payment.

## Key Business Recommendations

1. Prioritize converting **month-to-month customers** to annual contracts with a modest incentive - contract type is the single biggest churn driver.
2. Add a **first-90-days onboarding check-in** - new customers are the highest-risk group.
3. Review pricing/service quality for **fiber-optic** customers, who churn more than DSL customers at similar price points.

## Repository Structure

```
customer-churn-prediction/
├── data/
│   └── telco_customer_churn.csv
├── notebooks/
│   └── customer_churn_prediction.ipynb   
├── models/
│   └── churn_model.pkl                   
├── outputs/
│   └── *.png                            
├── src/
│   └── predict_new_customer.py          
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/customer_churn_prediction.ipynb
```

To score a new customer with the already-trained model:

```bash
python src/predict_new_customer.py
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter

## Limitations & Next Steps

- Single time-snapshot dataset; a production version would retrain on a rolling window and monitor for drift.
- Class imbalance handled via `class_weight='balanced'`; SMOTE or threshold tuning could be explored against the retention team's actual call capacity.
- Hyperparameter tuning and gradient boosting (XGBoost/LightGBM) are natural next steps for further performance gains.

## Author

Dhanasekaran T - [LinkedIn](https://www.linkedin.com/in/dhanasekaran-t1035/) · [GitHub](https://github.com/dhanasekaran65)
