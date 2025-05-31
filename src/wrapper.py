"""
Wrapper for the XGBoost model.
"""
import joblib
import pandas as pd
from dtos import OnePredictionOutputDto

FEATURES_TO_ENCODE: list[str] = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
                                 'distribution_channel', 'is_repeated_guest', 'reserved_room_type',
                                 'assigned_room_type', 'deposit_type', 'customer_type']


class LogisticRegressionModelWrapper:

    def __init__(self, model_path: str, label_encoder_path: str):
        """
        Args:
            model_path (str): model path.
            label_encoder_path (str): label encoder path.
        """
        # create the instance of the model
        self.model = self.load_model(model_path)

        # create the instance of the encoder
        self.label_encoder = self.load_label_encoder(label_encoder_path)


    @staticmethod
    def load_model(model_path: str):
        """
        Load the Logistic Regression Classification model

        Args:
            model_path (str): Path to the model.

        Returns:
            model: Pipeline with the Logistic Regression Classification model.

        Raise:
            RuntimeError: If there is an error loading the model.
        """

        try:
            # load the model
            model = joblib.load(model_path)

            return model
        except Exception:
            raise RuntimeError(f"Error loading the model.")


    @staticmethod
    def load_label_encoder(model_label_encoder_path: str):
        """
        Load the label encoder.

        Args:
            model_label_encoder_path (str): Path to the one hot encoder model.

        Returns:
            label_encoder: The loaded one hot encoder.

        Raise:
            RuntimeError: If there is an error loading the encoder.
        """

        try:
            # load the label encoder
            label_encoder = joblib.load(model_label_encoder_path)

            return label_encoder
        except Exception:
            raise RuntimeError(f"Error loading the encoder")


    @staticmethod
    def from_dict_to_df(features: dict) -> pd.DataFrame:
        """
        Transform a dictionary to a pandas dataframe.

        Args:
            features (dict): The input dictionary to transform.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """

        # representation of one row
        row: dict = {}

        # for each key and value in the dictionary, we add the key and value to the row
        for key, value in features.items():
            # we add the key and value to the row
            row[key] = value

        # transform to dataframe
        df: pd.DataFrame = pd.DataFrame([row])

        return df


    def predict(self, df: pd.DataFrame) -> (str, float):
        """
        Do the preprocessing and the predictions.

        Args:
            df (pd.DataFrame): The elements to do the prediction on.

        Returns:
            The made predictions.
        """

        try:
            # do the prediction, remember that everything is in the pipeline, then use the label encoder to
            # inverse transform
            pred = self.model.predict(df)
            pred = self.label_encoder.inverse_transform(pred)
            pred_proba = self.model.predict_proba(df)

            return pred, pred_proba

        except Exception:
            raise ValueError(f"Oops, error while doing the prediction.")


    def predict_one(self, data: dict) -> OnePredictionOutputDto:
        """
        Do the predictions for just one set of features.

        Args:
            data (dict): The features as dictionary to do the prediction.

        Returns:
            dict: The prediction as a dictionary.
        """

        try:
            # transform from dict to df
            df = self.from_dict_to_df(data)

            # do the predictions
            pred, pred_proba = self.predict(df)

            # create the output transform
            output: dict = {
                "prediction": pred.tolist()[0],
                "proba_0": pred_proba.tolist()[0][0],
                "proba_1": pred_proba.tolist()[0][1]
            }

            return output

        except Exception:
            raise ValueError(f"Oops, error while doing the prediction.")
