# Toxic Comment Classification with ⚡ Lightning and 🤗 Transformers

# Overview
Toxicity in language models relates to the use of offensive, vulgar, or otherwise inappropriate language, usually in the form of the model’s response to a prompt. Toxic outputs occur because models are trained on vast amounts of language data. Any toxic language, or otherwise harmful biases within the training data, is learned by the model. Since LLMs work by predicting the next most likely word in a sequence, they regularly output stereotypes based on the language they are trained on.

This repo provides model and code for fine-tuning a transformer model on a specific use case and a specific dataset. Specific use case is multi-label toxic classification and specific dataset is a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. 

Used in LLM Observability suite to detect and monitor potentially harmful or inappropriate content within LLM outputs.

# Prediction
1. git clone https://github.com/furqan-shaikh/toxic-classifier.git
2. Create python virtual environment and install requirements
3. Run:
   ```python
   from prediction_runner import run_prediction
   results = run_prediction(model_type="original_small",
                            comments=["i dont like you, you sucker","i like you"])
   # [
   #    [{'label': 'toxic', 'probability': 0.95, 'prediction': 1},
   #    {'label': 'severe_toxic', 'probability': 0.03, 'prediction': 0}, 
   #    {'label': 'obscene', 'probability': 0.53, 'prediction': 1},
   #    {'label': 'threat', 'probability': 0.01, 'prediction': 0},
   #    {'label': 'insult', 'probability': 0.82, 'prediction': 1},
   #    {'label': 'identity_hate', 'probability': 0.02, 'prediction': 0}],
   
   #    [{'label': 'toxic', 'probability': 0.01, 'prediction': 0},
   #    {'label': 'severe_toxic', 'probability': 0.0, 'prediction': 0}, 
   #    {'label': 'obscene', 'probability': 0.0, 'prediction': 0},
   #    {'label': 'threat', 'probability': 0.0, 'prediction': 0},
   #    {'label': 'insult', 'probability': 0.0, 'prediction': 0},
   #    {'label': 'identity_hate', 'probability': 0.0, 'prediction': 0}]
   # ]
   ```

# Training
1. Download data from Kaggle here
2. Under `config/training_toxic_classification_config.json`, make the following changes:
    - add path of the training csv under `data/training/path`
3. Create python virtual environment and install requirements
4. Run: `python training_runner.py`
5. At the end of the training, the fine-tuned model is available at the path provided in config json

## Architecture
![untoxify_architecture.png](docs/untoxify_architecture.png)

# What's pending
1. Model Evaluation
2. Multilingual Toxic Comment Classification
3. Unintended Bias in Toxicity Challenge
4. Monitor training progress with tensorboard