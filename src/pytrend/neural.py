import pandas as pd
import numpy as np
import joblib, os, shutil, datetime
import torch

import logging, gc

logger = logging.getLogger("Numerai")


from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import cupy as cp


# +
## Tabular Models
# -


class TabularModel:
    def __init__(self, nn_model, config):

        """
        Args:
            nn_model (LightningModule): Neural Networks implmented as a LightningModule
            config (dict): A dictionary which contains the parameters for training NN
        """

        self.nn_model = nn_model
        self.config = config
        seed_everything(config.get("seed", 0), workers=True)

    def train(self, X_train, y_train, X_validate, y_validate):

        self.config["input_shape"] = X_train.shape[1]
        self.config["output_shape"] = y_train.shape[1]

        self.network = self.nn_model(self.config)

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.config.get("patience", 5),
            verbose=False,
            mode="min",
        )

        ## Assume X is a DataFrame, assume y is a DataFrame or pd Series
        dataset_train = TensorDataset(
            torch.from_numpy(X_train.values), torch.from_numpy(y_train.values)
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=self.config.get("batch_size", 4096),
            num_workers=0,
        )
        dataset_validate = TensorDataset(
            torch.from_numpy(X_validate.values), torch.from_numpy(y_validate.values)
        )
        dataloader_validate = DataLoader(
            dataset_validate,
            batch_size=self.config.get("batch_size", 4096),
            num_workers=0,
        )

        ## Use GPU if possible
        self.trainer = Trainer(
            accelerator="cuda",
            deterministic=True,
            auto_lr_find=True,
            max_epochs=self.config.get("max_epochs", 3),
            callbacks=[early_stop_callback],
        )

        self.trainer.fit(self.network, dataloader_train, dataloader_validate)

    def predict(self, X):
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(torch.from_numpy(X))
        return predictions.numpy()

    def load_model(self, checkpoint):
        self.network = self.nn_model.load_from_checkpoint(checkpoint)

    def save_model(self, checkpoint):
        self.trainer.save_checkpoint(checkpoint)


# +
## Tabular Modules
# -


class MLP(LightningModule):
    def __init__(self, config):

        super().__init__()
        self.config = config

        neuron_sizes = config.get("neurons", 256)
        num_layers = config.get("num_layers", 2)

        self.layers = nn.Sequential(
            nn.Linear(config["input_shape"], neuron_sizes),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.5)),
        )

        for i in range(
            1,
            num_layers,
        ):
            new_neuron_sizes = int(neuron_sizes * config.get("neuron_scale", 0.5)) + 1
            self.layers.append(
                nn.Linear(neuron_sizes, new_neuron_sizes),
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(config.get("dropout", 0.5)))
            neuron_sizes = new_neuron_sizes

        self.layers.append(nn.Linear(neuron_sizes, config["output_shape"]))

        ## Need to have this to ensure correct hyper-parameters are loaded
        ## https://github.com/Lightning-AI/lightning/issues/3981
        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x.float())

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x.float())
        loss = F.mse_loss(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x.float())
        loss = F.mse_loss(y_hat, y.float())
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.get("learning_rate", 1e-4)
        )
        return optimizer


class LSTM_Tabular(LightningModule):
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=self.config.get("no_channels", 4),
            hidden_size=self.config.get("hidden_size", 4),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.1),
            batch_first=True,
        )
        self.fc = nn.Linear(
            self.config.get("hidden_size", 4), self.config.get("output_shape", 1)
        )
        self.save_hyperparameters()

    def forward(self, x):
        batch_size, flattened = x.shape
        x = torch.reshape(
            x,
            (
                batch_size,
                -1,
                self.config.get("no_channels", 4),
            ),
        )
        flip_columns_order = self.config.get("flip_column_order", True)
        if flip_columns_order:
            x = torch.flip(x, [1])
        ## LSTM Layers
        lstm_out, _ = self.lstm(
            x.float()
        )  # lstm_out = (batch_size, seq_len, hidden_size)
        x = self.fc(lstm_out[:, -1])
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat.float(), y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat.float(), y.float())
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# +
## Time Series Models
## Needs Further Development
# -


class TimeSeriesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, timeseries, targets=None, lookback=200):
        """
        Args:
            timeseries (pd.DataFrame): A DataFrame of a multivaraite time-series
            targets (pd.DataFrame): A DataFrame of targets for the time-series
            lookback (int): Number of data records to include in lookback
        """
        self.X = timeseries
        self.y = targets
        self.lookback = lookback

    def __len__(self):
        return self.X.shape[0] - (self.lookback - 1)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.y is not None:
            return torch.tensor(
                self.X.values[idx : idx + self.lookback, :]
            ), torch.tensor(self.y.values[idx + self.lookback - 1, :])
        else:
            return torch.tensor(self.X.values[idx : idx + self.lookback, :])


class TimeSeriesModel:
    def __init__(self, nn_model, config):

        """
        Args:
            nn_model (LightningModule): Neural Networks implmented as a LightningModule
            config (dict): A dictionary which contains the parameters for training NN
        """

        self.nn_model = nn_model
        self.config = config
        seed_everything(config.get("seed", 0), workers=True)

    def train(self, X_train, y_train, X_validate, y_validate):

        self.config["input_shape"] = X_train.shape[1]
        self.config["output_shape"] = y_train.shape[1]

        self.network = self.nn_model(self.config)

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.config.get("patience", 10),
            verbose=False,
            mode="min",
        )

        ## Assume X is a DataFrame, assume y is a DataFrame or pd Series

        train_dataset = TimeSeriesDataset(
            X_train, y_train, lookback=self.config.get("lookback", 200)
        )
        dataloader_train = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.get("batch_size", 1000), shuffle=False
        )
        validate_dataset = TimeSeriesDataset(
            X_validate, y_validate, lookback=self.config.get("lookback", 200)
        )
        dataloader_validate = torch.utils.data.DataLoader(
            validate_dataset,
            batch_size=self.config.get("batch_size", 1000),
            shuffle=False,
        )

        ## Use GPU if possible
        self.trainer = Trainer(
            accelerator="cuda",
            deterministic=True,
            auto_lr_find=True,
            max_epochs=self.config.get("max_epochs", 3),
            callbacks=[early_stop_callback],
        )

        self.trainer.fit(self.network, dataloader_train, dataloader_validate)

    def predict(self, X):
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(X)
        return predictions.numpy()

    def load_model(self, checkpoint):
        self.network = self.nn_model.load_from_checkpoint(checkpoint)

    def save_model(self, checkpoint):
        self.trainer.save_checkpoint(checkpoint)


# +
### TimeSeires Modules
# -


class LSTM(LightningModule):
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=self.config.get("input_size", 11),
            hidden_size=self.config.get("hidden_size", 4),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.1),
            batch_first=True,
        )
        self.fc = nn.Linear(self.config.get("hidden_size", 4), 11)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out = (batch_size, seq_len, hidden_size)
        x = self.fc(lstm_out[:, -1])
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        loss = F.mse_loss(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        loss = F.mse_loss(y_hat, y.float())
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
