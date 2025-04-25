from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from prediction_runner import run_prediction

app = FastAPI()

class Request(BaseModel):
    model_type: str
    comments: List[str]

@app.post("/predict")
async def predict(req: Request):
    return run_prediction(req.model_type, req.comments)
