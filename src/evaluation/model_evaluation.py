import pandas as pd
import os
import json
import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from logger.logger import get_logger

logger = get_logger('Model-Evaluation')


def load_data(x_path: str, y_path: str):
    """Load test features and labels"""
    try:
        logger.info(f"Loading test data from {x_path} and {y_path}")
        x_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path).squeeze()  # Convert to Series
        return x_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise


def load_model(model_path: str) -> Pipeline:
    """Load a trained model from pickle"""
    try:
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def evaluate_classification(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics"""
    try:
        y_pred = model.predict(x_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[:, 1]
        else:
            y_prob = y_pred  # fallback

        metrics = {
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "Test_AUC": roc_auc_score(y_test, y_prob)
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_metrics(metrics: dict, path: str):
    """Save metrics to JSON for DVC tracking"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise


def main():
    try:
        # Load test data
        x_test, y_test = load_data(
            "data/processed/split/X_test.csv",
            "data/processed/split/y_test.csv"
        )

        # Load model
        model = load_model("models/model.pkl")

        # Evaluate
        metrics = evaluate_classification(model, x_test, y_test)

        # Save metrics locally (DVC-ready)
        save_metrics(metrics, "reports/metrics.json")

        # Print summary
        print("=" * 50)
        print("🏆 Classification Model Evaluation")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key:<10}: {value:.4f}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()