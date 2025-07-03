import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from src.logger import logging
from src.exception import USvisaException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.utils.main_utils import load_numpy_array_data, save_object


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        try:
            logging.info("🔧 Initializing ModelTrainer")
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise USvisaException(e, sys)

    def train_model(self) -> ModelTrainerArtifact:
        try:
            logging.info("📥 Loading transformed training data from .npy")
            data = load_numpy_array_data(self.model_trainer_config.transformed_train_file_path)

            X = data[:, :-1]
            y = data[:, -1]

            logging.info("🔀 Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logging.info("🌲 Training RandomForestClassifier")
            model = RandomForestClassifier(n_estimators=100,
                                             max_depth=5,  
                                             min_samples_leaf=10,
                                                  random_state=42)
            model.fit(X_train, y_train)

            logging.info("🧪 Evaluating model on test set")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logging.info(f"✅ Accuracy: {accuracy:.4f}")
            logging.info(f"✅ ROC-AUC: {roc_auc:.4f}")

            # Save model
            model_path = self.model_trainer_config.model_path
            save_object(model_path, model)
            logging.info(f"📦 Model saved at: {model_path}")

            # Save test set for evaluation
            test_array_path = self.model_trainer_config.test_array_path
            np.save(test_array_path, np.c_[X_test, y_test])
            logging.info(f"🧪 Test array saved at: {test_array_path}")

            return ModelTrainerArtifact(
                model_path=model_path,
                test_array_path=test_array_path,
                accuracy=accuracy,
                roc_auc=roc_auc
            )

        except Exception as e:
            raise USvisaException(e, sys)

