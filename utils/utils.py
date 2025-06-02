"""
Utils functions for the project.
"""
import uuid
from src.wrapper import LogisticRegressionModelWrapper


def generate_uuid() -> str:
    """
    Generate a UUID string.
    """
    return str(uuid.uuid4())


def config_model(model_path: str, label_encoder_path: str) -> LogisticRegressionModelWrapper:
    """
    Configure a LogisticRegressionModelWrapper.

    Args:
        model_path:
        label_encoder_path:

    Returns:
        LogisticRegressionModelWrapper

    Raises:
        RuntimeError: If one of the model paths does not exists.
    """

    try:
        model_wrapper = LogisticRegressionModelWrapper(
            model_path="models/model.pkl",
            label_encoder_path="models/encoder.pkl"
        )

        return model_wrapper

    except RuntimeError as e:
        raise Exception(f"No model. Cannot be initialized {e}")