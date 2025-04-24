import os

from configuration import load_config
from training_module import run_training

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config/training_toxic_classification_config.json")
    config_data=load_config(config_path)
    run_training(config_data)

if __name__ == "__main__":
    main()