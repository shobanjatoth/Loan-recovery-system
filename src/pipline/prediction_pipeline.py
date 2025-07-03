# === prediction_pipeline.py ===

import os
import sys
import warnings
import pandas as pd

from src.utils.main_utils import load_object
from src.exception import USvisaException
from src.logger import logging


def get_latest_artifact_path(subdir_name: str) -> str:
    try:
        base_artifact_path = "artifact"
        subdirs = sorted(
            [d for d in os.listdir(base_artifact_path) if os.path.isdir(os.path.join(base_artifact_path, d))],
            reverse=True
        )
        if not subdirs:
            raise FileNotFoundError("No timestamped artifact directories found.")

        latest_dir = subdirs[0]
        full_path = os.path.join(base_artifact_path, latest_dir, subdir_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Expected path not found: {full_path}")

        return full_path
    except Exception as e:
        raise USvisaException(e, sys)


class PredictionPipeline:
    def __init__(self):
        try:
            # Automatically resolve paths to latest model and transformer
            self.model_path = os.path.join(get_latest_artifact_path("model_trainer"), "risk_classifier.pkl")
            self.transformer_path = os.path.join(get_latest_artifact_path("data_transformation"), "transformer.pkl")

            logging.info(f"📦 Loading model from: {self.model_path}")
            logging.info(f"📦 Loading transformer from: {self.transformer_path}")

            # Load the model and transformer objects
            self.model = load_object(self.model_path)
            self.transformer = load_object(self.transformer_path)

        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, input_data: dict) -> dict:
        try:
            logging.info("🚀 Starting prediction pipeline")

            # Convert the input dict to a DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply the same transformation as during training
            transformed_data = self.transformer.transform(input_df)

            # Suppress feature name warnings from sklearn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                risk_scores = self.model.predict_proba(transformed_data)[:, 1]
                predicted_flags = (risk_scores > 0.5).astype(int)

            result = {
                "Risk_Score": float(risk_scores[0]),
                "Predicted_High_Risk": int(predicted_flags[0])
            }

            logging.info(f"✅ Prediction complete. Result: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys)
