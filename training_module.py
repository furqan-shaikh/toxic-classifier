import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_types import ConfigData
from toxic_dataset import ToxicClassificationDataset


class ToxicClassifier(L.LightningModule):
    def __init__(self, config_data: ConfigData):
        super().__init__()
        self.save_hyperparameters()
        self.config_data = config_data
        self.tokenizer, self.model = config_data.model.tokenizer, config_data.model.model

    def training_step(self, batch, batch_idx):
        # get all the comments from the training batch
        comments, meta = batch
        # do a forward pass passing the comments which invokes the model and gets the model outputs
        output = self.forward(comments)
        # compute loss using loss function
        loss = self.binary_cross_entropy(output, meta)
        # log the loss
        self.log("train_loss", loss)
        return {"loss": loss}

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        outputs = self.model(**inputs)[0]
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.config_data.adam_optimizer.lr,                 # learning rate
                                weight_decay=self.config_data.adam_optimizer.weight_decay,       # L2 regularization (helps reduce overfitting)
                                amsgrad=self.config_data.adam_optimizer.amsgrad )            # variant of Adam with better convergence in some cases)

    def binary_cross_entropy(self, input, meta):
        return F.binary_cross_entropy_with_logits(input, meta["target"].float())

def run_training(config_data: ConfigData):
    # Load your dataset
    dataset = ToxicClassificationDataset(train_file=config_data.training_data.training_path, classes=config_data.classes)

    data_loader = DataLoader(
        dataset,
        batch_size=config_data.run_arguments.batch_size,
        shuffle=config_data.run_arguments.shuffle
    )
    # model
    model = ToxicClassifier(config_data)

    trainer = L.Trainer(
        max_epochs=config_data.run_arguments.epoch,
        accumulate_grad_batches=config_data.run_arguments.accumulate_grad_batches
    )
    trainer.fit(
        model=model,
        train_dataloaders=data_loader
    )
    trainer.save_checkpoint(f"{config_data.checkpoint.path}{config_data.checkpoint.name}")
