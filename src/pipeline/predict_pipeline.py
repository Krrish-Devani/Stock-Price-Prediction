import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            if hasattr(model, 'estimators_'):  # RandomForest/Ensemble model
                for estimator in model.estimators_:
                    if not hasattr(estimator, 'monotonic_cst'):
                        estimator.monotonic_cst = None
            elif not hasattr(model, 'monotonic_cst'):  # Single DecisionTree
                model.monotonic_cst = None

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(
            self,
            Open: float,
            High: float,
            Low: float,
            Volume: float
    ):
        self.Open = Open
        self.High = High
        self.Low = Low
        self.Volume = Volume

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Open": [self.Open],
                "High": [self.High],
                "Low": [self.Low],
                "Volume": [self.Volume],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.error("Error occurred while converting custom data to DataFrame")
            raise CustomException(e, sys)