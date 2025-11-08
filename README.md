# 🎓 AI Dropout Prediction Model

This repository contains the implementation of a machine learning pipeline to predict student dropout risk using a Random Forest classifier. It is part of the "AI for Software Engineering" course assignment focused on applying the AI Development Workflow.

## 📘 Project Overview

- **Problem**: Predict student dropout risk based on engagement and demographic data.
- **Model**: Random Forest Classifier
- **Metrics**: Precision, Recall
- **Deployment**: Flask API with Docker containerization

## 🧪 Dataset

- LMS activity logs (e.g., login frequency, assignment submissions)
- Student demographics (e.g., age, major, location)
- Academic performance records (e.g., GPA, attendance)

## ⚙️ Preprocessing

- Missing value imputation using mean/mode
- Feature scaling and one-hot encoding
- Train/Validation/Test split (70/15/15)

## 🔧 Model Development

- Model: `RandomForestClassifier` from scikit-learn
- Hyperparameters tuned:
  - `n_estimators`: Number of trees in the forest
  - `max_depth`: Maximum depth of each tree
- Data split:
  - 70% training
  - 15% validation
  - 15% test

## 📈 Evaluation

- **Precision**: Measures how many predicted dropouts were actual dropouts
- **Recall**: Measures how many actual dropouts were correctly predicted
- Confusion matrix included for performance visualization

## 🚀 Deployment

- Flask API endpoint `/predict` for real-time predictions
- Model serialized with `joblib`
- Docker-ready for scalable deployment
- Concept drift monitoring strategy included


## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

  # Train and save the model
  python src/model_development.py

  # Evaluate the model
  python src/evaluation_deployment.py

  # Launch the API
  python src/app.py

