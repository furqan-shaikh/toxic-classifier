import transformers

def get_model_instance(class_name: str, model_name: str, num_labels: int):
    return  getattr(transformers, class_name).from_pretrained(model_name, num_labels=num_labels)

def get_tokenizer_instance(class_name: str, tokenizer_name: str):
    return getattr(transformers, class_name).from_pretrained(tokenizer_name)
