import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
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
    
    # Define models and their hyperparameter grids for GridSearchCV
    models_and_params = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"classifier__C": [0.1, 1.0, 10.0]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"classifier__n_estimators": [50, 100], "classifier__max_depth": [None, 10]}
        }
    }
    
    mlflow.set_experiment("Heart_Disease_Prediction")
    os.makedirs("models", exist_ok=True)
    
    for model_name, mp in models_and_params.items():
        with mlflow.start_run(run_name=model_name):
            # Create pipeline
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mp["model"])])
            
            # Perform Hyperparameter Tuning
            grid_search = GridSearchCV(pipeline, mp["params"], cv=5, scoring='roc_auc')
            grid_search.fit(X_train, y_train)
            
            best_pipeline = grid_search.best_estimator_
            mlflow.log_metric("cv_roc_auc_mean", grid_search.best_score_)
            
            # Predict using the best tuned model
            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }
            
            # Track experiments and best parameters with MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", model_name)
            
            # Log all optimized hyperparameters discovered by GridSearchCV
            for param_name, param_value in grid_search.best_params_.items():
                mlflow.log_param(param_name, param_value)
                
            mlflow.sklearn.log_model(best_pipeline, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
            print(f"[{model_name}] Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Save the best reproducible model
            if model_name == "Random_Forest":
                joblib.dump(best_pipeline, "models/model.pkl")
                print("-> Random Forest packaged and saved to models/model.pkl")

if __name__ == "__main__":
    train_and_log()