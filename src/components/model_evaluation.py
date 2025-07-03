import os
import sys
import json
import numpy as np
import mlflow
import mlflow.sklearn
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
    def __init__(self, model_trainer_artifact, data_transformation_artifact, model_evaluation_config):
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.model_evaluation_config = model_evaluation_config

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            model = load_object(self.model_trainer_artifact.model_path)
            transformer = load_object(self.data_transformation_artifact.transformer_object_path)

            test_data = np.load(self.model_trainer_artifact.test_array_path, allow_pickle=True)
            X_test = test_data[:, :-1]
            y_test = test_data[:, -1].astype(int)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Save evaluation YAML
            evaluation_result = {
                "model_path": self.model_trainer_artifact.model_path,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "classification_report": report
            }

            os.makedirs(os.path.dirname(self.model_evaluation_config.report_file_path), exist_ok=True)
            write_yaml_file(self.model_evaluation_config.report_file_path, evaluation_result)

            # ✅ MLflow logging
            mlflow.log_metric("eval_accuracy", accuracy)
            mlflow.log_metric("eval_roc_auc", roc_auc)

            # Log report file
            report_path = os.path.join(os.path.dirname(self.model_evaluation_config.report_file_path), "classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            mlflow.log_artifact(report_path, artifact_path="evaluation")
            mlflow.log_artifact(self.model_evaluation_config.report_file_path, artifact_path="evaluation")

            return ModelEvaluationArtifact(
                report_file_path=self.model_evaluation_config.report_file_path,
                accuracy=accuracy,
                roc_auc=roc_auc
            )

        except Exception as e:
            raise USvisaException(e, sys)



