import os
import requests
import time
import json
from dataclasses import dataclass
from typing import List
import torch
from training_module import ToxicClassifier

@dataclass
class PredictionConfig:
    base_download_url: str
    original_small: str
    cache_dir: str
    classes: List[str]
    original_large: str = ""

def load_config(config_path: str):
    with open(os.path.join(os.path.dirname(__file__), config_path)) as f:
        return _build_config_data(json.load(f))

def _build_config_data(config):
    return PredictionConfig(
        base_download_url= config["base_download_url"],
        original_small= config["original_small"],
        cache_dir=config["cache_dir"],
        classes=config["classes"]
    )

def run_prediction(model_type:str, comments=List[str]):
    """
    Run prediction based on the model type for the provided comments. The valid model_type values are:
    - original_small
    """
    config = load_config(config_path= "config/config.json")
    model = _get_model(config, model_type)
    # always call before doing inference / prediction.
    model.eval()
    with torch.no_grad():
        # do prediction on the model
        logits = model(comments)
        # convert logits to probabilities
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).int()
    results = []
    for prob, pred in zip(probs, predictions):
        res = []
        for label, probability, prediction in zip(config.classes, prob, pred):
            res.append({
                "label": label,
                "probability":  round(probability.item(), 2),
                "prediction": prediction.item()
            })
        results.append(res)
    return results

def _download_model(config: PredictionConfig, model_type: str):
    os.makedirs(os.path.expanduser(config.cache_dir), exist_ok=True)
    filename = os.path.basename(config.original_small)
    local_path = os.path.join(os.path.expanduser(config.cache_dir), filename)
    model_url = ""
    if model_type == "original_small":
        model_url = f"{config.base_download_url}{config.original_small}"
    

    if not os.path.exists(local_path):
        print(f"Downloading model from {model_url} to {local_path}...")
        response = requests.get(model_url)
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded model from {model_url} to {local_path}...")
    else:
        print(f"Using cached model from {local_path}")

    return local_path

def _load_model(model_file: str):
    start_time = time.time()
    # load the model
    model = ToxicClassifier.load_from_checkpoint(model_file)
    print("-Model loaded in -- %s seconds ---" % (time.time() - start_time))
    return model

def _get_model(config, model_type):
    model_path = _download_model(config, model_type)
    model = _load_model(model_path)
    return model