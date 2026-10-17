import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def train_and_log():
    # Load the preprocessed dataset
    df = pd.read_csv("data/heart_disease.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Engineering
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    
    # Define models to train[cite: 1]
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    mlflow.set_experiment("Heart_Disease_Prediction")
    os.makedirs("models", exist_ok=True)
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Create a full reproducible pipeline[cite: 1]
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
            
            # Cross-validation[cite: 1]
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            
            # Train and Predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate required metrics[cite: 1]
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }
            
            # Track experiments with MLflow[cite: 1]
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", model_name)
            mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
            print(f"[{model_name}] Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Save the final reproducible model format for the API[cite: 1]
            if model_name == "Random_Forest":
                joblib.dump(pipeline, "models/model.pkl")
                print("-> Random Forest packaged and saved to models/model.pkl")

if __name__ == "__main__":
    train_and_log()