# AI Accounting Fraud Detection

An end-to-end machine learning project that classifies accounting records into four categories using a real-world research dataset from Harvard Dataverse.

## Overview

This project uses a trained Random Forest classifier to analyze accounting-related features and predict one of four categories. It includes:

- Multi-class fraud classification
- Prediction confidence scoring
- Feature importance analysis
- Interactive Streamlit dashboard
- Record-level inspection

## Dataset

Source: Harvard Dataverse  
Dataset: **Accounting Fraud**  
Author: Peifeng Wu (2024)

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- OpenPyXL
- Joblib

## Results

The Random Forest model achieved **97% accuracy** on the test set.

### Classification Performance
- Macro F1-score: **0.97**
- Weighted F1-score: **0.97**

### Top Important Features
1. **Feature 5** — 0.2666
2. **Feature 8** — 0.1861
3. **Feature 10** — 0.1475
4. **Feature 3** — 0.1107
5. **Feature 9** — 0.0753

These results suggest that the model can reliably classify accounting records into four categories while also providing interpretable feature-level insight into its predictions.

### Top Important Features
- Feature 5
- Feature 8
- Feature 10
- Feature 3
- Feature 9

## Project Structure

```text
ai-accounting-fraud-detection/
├── app/
│   └── dashboard.py
├── data/
│   └── data.xlsx
├── models/
│   ├── train_model.py
│   ├── feature_importance.py
│   └── fraud_model.pkl
├── notebooks/
│   └── explore_data.py
├── rag/
├── utils/
├── README.md
├── requirements.txt
└── .gitignore
