
from typing import Union, List
import os
import requests
import time
import json
from dataclasses import dataclass

import torch
from transformers import AlbertTokenizer

from training_module import ToxicClassifier
@dataclass
class PredictionConfig:
    base_download_url: str
    original_small: str
    cache_dir: str
    original_large: str = ""

def load_config(config_path: str):
    with open(config_path) as f:
        return _build_config_data(json.load(f))

def _build_config_data(config):
    return PredictionConfig(
        base_download_url= config["base_download_url"],
        original_small= config["original_small"],
        cache_dir=config["cache_dir"]
    )

def run_prediction(model_type:str, comments=Union[str, List[str]]):
    """
    Run prediction based on the model type for the provided comments. The valid model_type values are:
    - original_small
    """
    config_path = os.path.join(os.path.dirname(__file__), "config/config.json")
    config = load_config(config_path)
    model_path = _download_model(config, model_type)
    model = _load_model(model_path)
    tokenizer = _load_tokenizer(model_type)

    # # Tokenize the input comment
    inputs = tokenizer(comments, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        # do prediction on the model
        logits = model(comments)
        # convert logits to probabilities
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).int()
    for label, prob, pred in zip(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], probs[0], predictions[0]):
        print(f"{label:15} | prob: {prob:.2f} | pred: {pred.item()}")

def _download_model(config: PredictionConfig, model_type: str):
    os.makedirs(os.path.expanduser(config.cache_dir), exist_ok=True)
    filename = os.path.basename(config.original_small)
    local_path = os.path.join(os.path.expanduser(config.cache_dir), filename)
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

def _load_tokenizer(model_type: str):
    # load the tokenizer
    if model_type == "original_small":
        return AlbertTokenizer.from_pretrained("albert-base-v2")


run_prediction(model_type="original_small", comments="i dont like you, you sucker")
# run_prediction(model_file="training_toxic_run_1.ckpt", comment="Fuck off, you anti-semitic cunt.")

