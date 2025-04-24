import json

from model_tokenizer import get_model_instance, get_tokenizer_instance
from model_types import TrainingData, AdamOptimizer, RunArguments, Checkpoint, ConfigData, Model


def load_config(config_path: str):
    with open(config_path) as f:
        return _build_config_data(json.load(f))

def _build_config_data(config):
    current_model = config["current_model"]
    training_data = TrainingData(
        training_path=config["data"]["training"]["path"]
    )
    run_arguments = RunArguments(
        batch_size=int(config["run_args"]["batch_size"]),
        shuffle=config["run_args"]["shuffle"],
        epoch=int(config["run_args"]["epoch"]),
        accumulate_grad_batches=int(config["run_args"]["accumulate_grad_batches"])
    )
    adam_optimizer = AdamOptimizer(
        lr=float(config["adam_optimizer"]["lr"]),
        weight_decay=float(config["adam_optimizer"]["weight_decay"]),
        amsgrad=bool(config["adam_optimizer"]["amsgrad"])
    )
    checkpoint = Checkpoint(
        name=config["checkpoint"]["name"],
        path=config["checkpoint"]["path"]
    )

    return ConfigData(
        adam_optimizer= adam_optimizer,
        checkpoint= checkpoint,
        classes= config["classes"],
        model= _get_model(config, current_model),
        run_arguments= run_arguments,
        training_data= training_data
    )

def _get_model(config, current_model):
    models = config["models"]
    for model in models:
        if model["model_key"] == current_model:
            return Model(
                model_type= model["model_type"],
                model= get_model_instance(class_name=model["model_name"], model_name=model["model_type"], num_labels=len(config["classes"])),
                tokenizer= get_tokenizer_instance(class_name=model["tokenizer"], tokenizer_name=model["model_type"])
            )
                
            