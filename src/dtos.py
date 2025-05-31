from pydantic import BaseModel


class OnePredictionInputDto(BaseModel):
    """
    Input DTO features to make a prediction.
    """

    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class OnePredictionOutputDto(BaseModel):
    """
    Output DTO for one prediction.
    """

    prediction: str
    proba_0: float
    proba_1: float
