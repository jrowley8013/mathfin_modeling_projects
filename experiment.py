"""Model training and experiment configuration."""

import time
from dataclasses import dataclass
from typing import Literal, Optional
from zlib import adler32

import numpy as np
import os
import pandas as pd
import torch
import yaml
from data import Batch, USE_GPU
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from model import make_lstm_model, make_model, rate

EXPERIMENTS_DIR = Path("modeling_experiments/")


@dataclass
class EpochTrainingMetric:
    """Epoch training metric class."""

    loss: float
    token_rate: int
    lr: float
    elapsed: float
    annualized_sharpe: Optional[float] = None


@dataclass
class TrainingConfig:
    """Training configurtion class."""

    batch_size: int
    n_assets: int  # number of assets
    tau: int  # time-index length of panel data

    h: int  # number of attention heads
    N: int  # number of encoder/decoder layers
    d_model: int  # model embedding dimension
    dropout: float  # dropout rate

    conv_kernel_size: int  # Conv1D kernel size
    padding: int  # padding for Conv1D layer

    learning_rate: float
    epochs: int
    optimizer: str

    seed: int

    model_type: Literal["lstm", "transformer"]
    data_type: Literal["etf", "stocks"]

    def __post_init__(self) -> None:
        """Set the seed."""
        torch.manual_seed(self.seed)

    @property
    def output_seq_length(self) -> int:
        """Get the output sequence length, assuming encoder input == decoder input."""
        return int((self.tau - 1) / 2)

    def make_model(self) -> nn.Module:
        """Make the model from the configuration."""
        if self.model_type == "transformer":
            model = make_model(
                src_vocab=self.n_assets,
                tgt_vocab=self.n_assets,
                kernel_size=self.conv_kernel_size,
                padding=self.padding,
                N=self.N,
                d_model=self.d_model,
                h=self.h,
                output_seq_length=self.output_seq_length,
                dropout=self.dropout,
            )
        elif self.model_type == "lstm":
            model = make_lstm_model(
                d_model=self.d_model,
                n_assets=self.n_assets,
                kernel_size=self.conv_kernel_size,
                padding=self.padding,
                output_seq_length=self.tau - 1,
                hidden_size=self.h,
                num_lstm_layers=self.N,
            )
        return model


class EarlyStopping:
    """Early stopping class."""

    def __init__(self, patience: int = 20):
        """Initialize the early stopping class."""
        self.patience = patience
        self.counter = 0

    def __call__(self, val_score: float, running_max: float) -> bool:
        """Call the early stopping class."""
        if val_score < running_max:
            self.counter += 1
            # print(self.counter)
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
        return False


@dataclass
class PortfolioModelingExperiment:
    """Portfolio modeling experiment class."""

    model: nn.Module

    training_config: TrainingConfig

    training_data: pd.DataFrame
    validation_data: pd.DataFrame
    early_stopping_patience: Optional[int] = None

    @property
    def get_training_data(self) -> DataLoader:
        """Get the training DataLoader."""
        price_block_arrs = [
            w.to_numpy(dtype="float32").T
            for w in self.training_data.rolling(window=self.training_config.tau)
            if w.shape[0] == self.training_config.tau
        ]
        rolling_price_block_arrs = []
        for i in range(len(price_block_arrs) - self.training_config.batch_size):
            rolling_price_block_arrs += price_block_arrs[
                i : i + self.training_config.batch_size
            ]
        data_loader = DataLoader(
            rolling_price_block_arrs, batch_size=self.training_config.batch_size
        )
        return data_loader

    @property
    def get_validation_data(self) -> DataLoader:
        """Get the validation DataLoader."""
        price_block_arrs_val = [
            w.to_numpy(dtype="float32").T
            for w in self.validation_data.rolling(window=self.training_config.tau)
            if w.shape[0] == self.training_config.tau
        ]
        data_loader_val = DataLoader(price_block_arrs_val)
        return data_loader_val

    @property
    def hash_dir(self) -> str:
        """Get the hash directory."""
        exp_training_dict = self.training_config.__dict__
        sorted_keys = np.sort([k for k in exp_training_dict.keys()])
        sorted_values = [str(exp_training_dict[k]) for k in sorted_keys]
        return str(adler32("".join(sorted_values).encode("utf-8")))

    def maybe_check_early_stopping(
        self,
        annualized_sharpe: float,
        running_max: float,
        early_stopping: EarlyStopping,
    ) -> bool:
        """Perform an early-stopping check if the early_stopping_patience is set."""
        if self.early_stopping_patience:
            return early_stopping(val_score=annualized_sharpe, running_max=running_max)
        else:
            return False

    def run(
        self, validate: bool, store_model: bool = False, model_store_interval: int = 5
    ) -> None:
        """Run the experiment."""
        torch.manual_seed(self.training_config.seed)
        if USE_GPU:
            self.model.to(0)
        if self.training_config.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.training_config.learning_rate
            )
        elif self.training_config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.training_config.learning_rate
            )
        self.lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, model_size=self.training_config.d_model, factor=10.0, warmup=80
            ),
        )
        running_max = -np.inf
        if self.early_stopping_patience:
            early_stopping = EarlyStopping(patience=self.early_stopping_patience)
        for epoch_idx in range(self.training_config.epochs):
            print(f"Epoch {epoch_idx} of {self.training_config.epochs}", end="")
            epoch_training_metric = self.run_epoch(
                optimizer=optimizer, validate=validate
            )
            if self.maybe_check_early_stopping(
                annualized_sharpe=epoch_training_metric.annualized_sharpe,
                running_max=running_max,
                early_stopping=early_stopping,
            ):
                print(
                    "Out-of-sample performance not improving."
                    + "Stopping training early,"
                )
                break
            else:
                running_max = max(running_max, epoch_training_metric.annualized_sharpe)

            if store_model and epoch_idx % model_store_interval == 0:
                self.store(
                    epoch_idx=epoch_idx, epoch_training_metric=epoch_training_metric
                )

    def store(self, epoch_idx: int, epoch_training_metric: EpochTrainingMetric) -> None:
        """Store the current state of the experiment."""
        training_info_dict = self.training_config.__dict__
        epoch_info_dict = {"epoch_index": epoch_idx, **epoch_training_metric.__dict__}
        info_dict = {**training_info_dict, **epoch_info_dict}

        model_experiment_dir = EXPERIMENTS_DIR / self.hash_dir

        os.makedirs(model_experiment_dir, exist_ok=True)

        with open(model_experiment_dir / f"epoch={epoch_idx}.yaml", "w+") as f:
            yaml.dump(info_dict, f)

        model_state = self.model.state_dict()
        torch.save(
            model_state,
            model_experiment_dir / f"epoch={epoch_idx}_model_state.pt",
        )

    def compute_negsharpe(self, out, returns):
        """Compute the negative Sharpe ratio."""
        weights = self.model.generator(out)
        port_returns = (weights.multiply(returns)).sum(dim=1)
        port_returns_mean = port_returns.mean()
        port_returns_std = port_returns.std()
        sharpe_ratio = -port_returns_mean / port_returns_std
        return sharpe_ratio.data, sharpe_ratio

    def run_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        validate: bool = False,
    ) -> EpochTrainingMetric:
        """Run the training epoch."""
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        for i, df_batch in enumerate(self.get_training_data):
            df_batch = df_batch.transpose(1, 2)
            batch = Batch(df_batch=df_batch)

            if self.training_config.model_type == "lstm":
                out = self.model.forward(batch.srctgt)[0]
            elif self.training_config.model_type == "transformer":
                out = self.model.forward(
                    batch.src,
                    batch.tgt,
                    batch.src_mask,
                    batch.tgt_mask,
                )[0]

            loss, loss_node = self.compute_negsharpe(out, batch.last_return)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            n_accum += 1
        epoch_avg_loss = total_loss / (i + 1)
        del loss
        del loss_node
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        annualized_sharpe = None

        valid_addendum = ""
        if validate:
            self.model.eval()

            rets = []
            with torch.no_grad():
                for i, data in enumerate(self.get_validation_data):
                    batch = Batch(df_batch=data.transpose(1, 2))
                    if self.training_config.model_type == "lstm":
                        out = self.model.forward(batch.srctgt)[0]
                    elif self.training_config.model_type == "transformer":
                        out = self.model.forward(
                            batch.src,
                            batch.tgt,
                            batch.src_mask,
                            batch.tgt_mask,
                        )[0]
                    # out = self.model.forward(batch.srctgt)[0]
                    weights = self.model.generator(out)
                    port_return = (weights * batch.last_return).sum(dim=1).cpu()
                    rets.append(port_return.numpy().item())

            annualized_sharpe = np.sqrt(252) * (
                pd.Series(rets).mean() / pd.Series(rets).std()
            )
            valid_addendum = f" | Annualized Sharpe: {annualized_sharpe:6.5f}"
            self.model.train()

        print(
            (
                (
                    " | Loss: %6.5f | Sec: %7.1f | Tokens / Sec: %7.1f"
                    + " | Learning Rate: %6.1e"
                )
                % (epoch_avg_loss, elapsed, tokens / elapsed, lr)
            )
            + valid_addendum
        )
        start = time.time()
        tokens = 0

        return EpochTrainingMetric(
            loss=epoch_avg_loss.cpu().numpy().item(),
            elapsed=elapsed,
            token_rate=tokens / elapsed,
            lr=lr,
            annualized_sharpe=annualized_sharpe.item(),
        )
