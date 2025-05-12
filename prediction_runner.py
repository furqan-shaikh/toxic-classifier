import os
import requests
import time
import json
from dataclasses import dataclass, field
from typing import List
import torch
from training_module import ToxicClassifier
from model_tokenizer import get_model_instance, get_tokenizer_instance

@dataclass
class ModelTokenizerConfig:
    model_key: str
    model_type: str
    model_name: str
    tokenizer: str

@dataclass
class PredictionConfig:
    base_download_url: str
    original_small: str
    cache_dir: str
    classes: List[str]
    original_large: str = ""
    base_model_tokenizer: dict[str,ModelTokenizerConfig ] = field(default_factory=dict)

def load_config(config_path: str):
    with open(os.path.join(os.path.dirname(__file__), config_path)) as f:
        return _build_config_data(json.load(f))

def _build_config_data(config):
    return PredictionConfig(
        base_download_url= config["base_download_url"],
        original_small= config["original_small"],
        cache_dir=config["cache_dir"],
        classes=config["classes"],
        base_model_tokenizer=_load_base_model_tokenizer_config(config)
    )

def _load_base_model_tokenizer_config(config):
    base_model_tokenizer_config=config["base_model_tokenizer"]
    base_model_tokenizer = {}
    for key,value in base_model_tokenizer_config.items():
        base_model_tokenizer[key] = ModelTokenizerConfig(
            model_key=value["model_key"],
            model_type=value["model_type"],
            model_name=value["model_name"],
            tokenizer=value["tokenizer"],
        )
    return base_model_tokenizer

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
    model_url = _load_base_model_tokenizer(config=config, model_type=model_type)
    
    if not os.path.exists(local_path):
        print(f"Downloading model from {model_url} to {local_path}...")
        response = requests.get(model_url)
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded model from {model_url} to {local_path}...")
    else:
        print(f"Using cached model from {local_path}")

    return local_path

def _load_base_model_tokenizer(config: PredictionConfig, model_type: str) -> str:
    model_url = f"{config.base_download_url}{config.original_small}"

    # load the model and tokenizer if not already available
    base_model_tokenizer_config = config.base_model_tokenizer[model_type]
    model= get_model_instance(class_name=base_model_tokenizer_config.model_name, model_name=base_model_tokenizer_config.model_type, num_labels=len(config.classes)),
    tokenizer= get_tokenizer_instance(class_name=base_model_tokenizer_config.tokenizer, tokenizer_name=base_model_tokenizer_config.model_type)

    return model_url


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