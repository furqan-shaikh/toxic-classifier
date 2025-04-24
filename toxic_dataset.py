import os
import pandas as pd
from torch.utils.data import Dataset
import torch

class ToxicClassificationDataset(Dataset):
    """
    A custom Dataset class must implement three functions: __init__, __len__, and __getitem__
    Retrieves our dataset’s features and labels one sample at a time
    Dataset comprises of following columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
    """
    def __init__(self, train_file, classes):
        self.train_data_frame = pd.read_csv(train_file)
        self.classes = classes

    def __len__(self):
        return len(self.train_data_frame)

    def __getitem__(self, idx):
        # get the row based on passed index from pandas data frame
        row = self.train_data_frame.iloc[idx]
        # grab the comment
        text = row["comment_text"]
        # grab the id
        text_id = row["id"]
        # set meta
        meta = {}
        meta["target"] = torch.tensor(list({label: value for label, value in row[self.classes].items()}.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return text, meta

        