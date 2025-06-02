import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from dtos import OnePredictionInputDto, OnePredictionOutputDto
from utils.utils import config_model


# load environment variables
load_dotenv()
cors_url = os.getenv("CORS_URL")

# load the ML model
model_wrapper = config_model(
    model_path="models/model.pkl",
    label_encoder_path="models/encoder.pkl"
)

app = FastAPI()

# CORS middleware configuration
origins = [
    cors_url,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "Hi there! I'm classification API. I can help you choose what crop to plant."}


@app.post("/predict")
def predict(features_dto: OnePredictionInputDto) -> OnePredictionOutputDto:
    """
    Endpoint for doing one prediction.

    Args:
        features_dto (FeaturesDto): The features to do a prediction.
    """

    try:
        # get the data from the request as a dictionary
        features: dict = features_dto.model_dump()

        result = model_wrapper.predict_one(features)

        return result
    except:
        raise HTTPException(
            status_code=500,
            detail="Error doing the prediction."
        )
