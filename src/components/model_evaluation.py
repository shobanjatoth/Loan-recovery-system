import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.logger import logging
from src.exception import USvisaException
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact
)
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_object, write_yaml_file


class ModelEvaluation:
    def __init__(self,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_evaluation_config: ModelEvaluationConfig):
        try:
            logging.info("🔧 Initializing ModelEvaluation component")
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("📥 Loading model and transformer")
            model = load_object(self.model_trainer_artifact.model_path)
            transformer = load_object(self.data_transformation_artifact.transformer_object_path)

            logging.info("📥 Loading test data from: %s", self.model_trainer_artifact.test_array_path)
            test_data = np.load(self.model_trainer_artifact.test_array_path, allow_pickle=True)
            X_test = test_data[:, :-1]
            y_test = test_data[:, -1].astype(int)

            logging.info("🤖 Predicting and scoring the model")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)

            logging.info(f"✅ Accuracy: {accuracy:.4f}")
            logging.info(f"✅ ROC AUC: {roc_auc:.4f}")

            # Create result dictionary
            evaluation_result = {
                "model_path": self.model_trainer_artifact.model_path,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "classification_report": report
            }

            # Write to YAML
            os.makedirs(os.path.dirname(self.model_evaluation_config.report_file_path), exist_ok=True)
            write_yaml_file(self.model_evaluation_config.report_file_path, evaluation_result)

            logging.info(f"📄 Evaluation report saved at: {self.model_evaluation_config.report_file_path}")

            return ModelEvaluationArtifact(
                report_file_path=self.model_evaluation_config.report_file_path,
                accuracy=accuracy,
                roc_auc=roc_auc
            )

        except Exception as e:
            raise USvisaException(e, sys)


