"""
util.py

Utility Files for training algorithms and data processing

Written by Will Solow, 2025
"""

import time
import os
from argparse import Namespace
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import _reduction as _Reduction
import torch.nn.functional as F
import numpy as np
from huggingface_hub import upload_folder
from omegaconf import OmegaConf, DictConfig

from model_engine.util import PHENOLOGY_INT


def set_embedding_op(model: nn.Module, embed_op: str) -> int:
    """
    Set the embedding operation to be used
    in MultiTask Embedding Models
    """
    if embed_op == "concat":

        def concat(embed, input):
            return torch.concatenate((embed, input), dim=-1)

        model.embed_op = concat
        return 2 * model.input_dim
    elif embed_op == "add":

        def add(embed, input):
            return embed + input

        model.embed_op = add
        return model.input_dim
    elif embed_op == "mult":

        def mult(embed, input):
            return embed * input

        model.embed_op = mult
        return model.input_dim


def load_config(fpath: str) -> DictConfig:
    """
    Loads configuration file of model
    """
    config = OmegaConf.load(f"{os.getcwd()}/{fpath}/config.yaml")

    return config


def load_model_fpath(fpath: str, name: str = "rnn_model_best.pt") -> str:

    return f"{os.getcwd()}/{fpath}/{name}"


def get_grad_norm(model: nn.Module) -> float:
    """
    Get gradients of model
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def compute_RMSE_STAGE_tensor(
    true_output: torch.Tensor,
    model_output: torch.Tensor,
    mask: torch.Tensor = None,
    stage: int = 0,
) -> torch.Tensor:
    """
    Compute the RMSE of a stage
    """
    curr_stage = (stage) % len(PHENOLOGY_INT)
    if mask is None:
        mask = torch.ones(len(true_output))

    true_stage_args = torch.argwhere((true_output == curr_stage) * mask).flatten()
    model_stage_args = torch.argwhere((model_output == curr_stage) * mask).flatten()

    if len(true_stage_args) == 0 or len(model_stage_args) == 0:
        return (len(true_stage_args) + len(model_stage_args)) ** 2
    else:
        return (true_stage_args[0] - model_stage_args[0]) ** 2


def cumulative_error(
    true: torch.Tensor,
    model: torch.Tensor,
    mask: torch.Tensor = None,
    n_stages: int = 5,
    RMSE: bool = True,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """
    Plot the errors in predicting onset of stage
    """
    avgs = torch.zeros(size=(n_stages,)).to(device)
    if model.size(-1) == 3:
        return avgs

    if model.size(-1) == len(PHENOLOGY_INT):  # Handle categorical classification
        probs = F.softmax(model, dim=-1)
        model = torch.argmax(probs, dim=-1)
    model = torch.round(model)  # Round to nearest integer for comparison
    if mask is None:
        mask = torch.ones(shape=true.shape).to(device)
    for s in range(n_stages):
        for i in range(len(true)):
            if RMSE:
                avgs[s] += compute_RMSE_STAGE_tensor(
                    true[i].flatten(),
                    model[i].flatten(),
                    mask=mask[i].flatten(),
                    stage=s,
                )
    return avgs


def setup_logging(args: Namespace) -> tuple[SummaryWriter, str]:
    """Setup Tensorboard Logging and W&B"""

    run_name = f"{args.run_name}__{int(time.time())}"
    log_path = f"{os.getcwd()}{args.log_path}{args.cultivar}/{run_name}"
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(args).items()])),
    )
    return writer, run_name, log_path


def log_training(
    calibrator: nn.Module,
    writer: SummaryWriter,
    fpath: str,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    train_avg: float,
    eval_avg: float,
    grad: float,
) -> None:
    """
    Log training statistics and print to console
    """

    # RMSE
    eval_len = len(calibrator.data["val"]) if calibrator.config.val_set else len(calibrator.data["test"])
    train_avg[:3] = torch.sqrt(train_avg[:3] / len(calibrator.data["train"]))
    eval_avg[:3] = torch.sqrt(eval_avg[:3] / eval_len)
    train_avg[-1] = torch.sum(train_avg[:3])
    eval_avg[-1] = torch.sum(eval_avg[:3])

    if hasattr(calibrator, "nn"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.nn.parameters()))
    elif hasattr(calibrator, "finetuner"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.finetuner.parameters()))

    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("eval_loss", eval_loss, epoch)
    for k in range(4):
        writer.add_scalar(f"train_rmse_{k}", np.round(train_avg[k].cpu().numpy(), decimals=2), epoch)
        writer.add_scalar(f"eval_rmse_{k}", np.round(eval_avg[k].cpu().numpy(), decimals=2), epoch)
    writer.add_scalar("model_grad_norm", grad / len(calibrator.data["train"]), epoch)
    writer.add_scalar("learning_rate", calibrator.optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("weight_norm", weight_norm, epoch)

    if calibrator.config.dtype == "grape_phenology":
        if calibrator.config.DConfig.loss_func == "BudbreakMSELoss":
            best_avg = eval_avg[0]
        elif calibrator.config.DConfig.loss_func == "BloomMSELoss":
            best_avg = eval_avg[1]
        elif calibrator.config.DConfig.loss_func == "VeraisonMSELoss":
            best_avg = eval_avg[2]
        else:
            best_avg = eval_avg[-1]
        if calibrator.best_cum_rmse > best_avg:
            calibrator.best_eval_loss = eval_loss
            calibrator.best_cum_rmse = best_avg
            calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")
            calibrator.best_rmse = eval_avg
    elif calibrator.config.dtype == "grape_coldhardiness":
        if calibrator.best_eval_loss > eval_loss:
            calibrator.best_eval_loss = eval_loss
            calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")
    elif calibrator.config.dtype == "wofost":
        if calibrator.best_eval_loss > eval_loss:
            calibrator.best_eval_loss = eval_loss
            calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")

    # calibrator.save_model(f"{fpath}", name="rnn_model_current.pt")

    p_str = f"############### Epoch {epoch} ###############\n"
    p_str += f"Train loss: {train_loss}\n"
    p_str += f"Val loss: {eval_loss}\n"
    p_str += f"Model Grad Norm: {grad/len(calibrator.data['train'])}\n"
    if calibrator.config.dtype == "grape_phenology":
        p_str += f"Train RMSE: {np.round(train_avg.cpu().numpy(),decimals=2)}\n"
        p_str += f"Val RMSE: {np.round(eval_avg.cpu().numpy(),decimals=2)}\n"
        p_str += f"Best Val RMSE: {np.round(calibrator.best_rmse.cpu().numpy(),decimals=2)}\n"

    print(p_str)


def log_error_training(
    calibrator: nn.Module,
    writer: SummaryWriter,
    fpath: str,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    train_avg: float,
    eval_avg: float,
    grad: float,
) -> None:
    """
    Log training statistics and print to console
    """

    if hasattr(calibrator, "nn"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.nn.parameters()))
    elif hasattr(calibrator, "finetuner"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.finetuner.parameters()))

    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("eval_loss", eval_loss, epoch)
    for k in range(4):
        writer.add_scalar(f"train_rmse_{k}", np.round(train_avg[k].cpu().numpy(), decimals=2), epoch)
        writer.add_scalar(f"eval_rmse_{k}", np.round(eval_avg[k].cpu().numpy(), decimals=2), epoch)
    writer.add_scalar("model_grad_norm", grad / len(calibrator.data["train"]), epoch)
    writer.add_scalar("learning_rate", calibrator.optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("weight_norm", weight_norm, epoch)

    if calibrator.best_eval_loss > eval_loss:
        calibrator.best_eval_loss = eval_loss
        calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")

    # calibrator.save_model(f"{fpath}", name="rnn_model_current.pt")

    p_str = f"############### Epoch {epoch} ###############\n"
    p_str += f"Train loss: {train_loss}\n"
    p_str += f"Val loss: {eval_loss}\n"
    p_str += f"Model Grad Norm: {grad/len(calibrator.data['train'])}\n"

    print(p_str)


def save_and_upload_hf(
    step: int,
    local_dir: str = "./_runs/",
    repo_id: str = "wsolow/ParamCalibration",
) -> None:
    """
    Upload a checkpointed model to hugging face
    """
    """repo = Repository(local_dir=local_dir, clone_from=repo_id)

    # Commit and push to HuggingFace
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(f"Upload checkpoint at step {step}")
    repo.git_push()"""

    upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        repo_type="model",
        commit_message=f"Checkpoint upload {step}",
    )


class StageMSELoss(nn.Module):
    """
    Stage MSE Loss function
    """

    __constants__ = ["reduction"]

    def __init__(self, stage: str, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.stage = stage

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mask losses based on only current stage loss"""
        input_mask = torch.round(input) == PHENOLOGY_INT[self.stage]
        target_mask = target == PHENOLOGY_INT[self.stage]
        mask = input_mask | target_mask

        input = torch.where(mask, input, 0.0)
        target = torch.where(mask, target, 0.0)

        return F.mse_loss(input, target, reduction=self.reduction)


class StageCrossEntropyLoss(nn.Module):
    __constants__ = ["ignore_index", "reduction", "label_smoothing"]
    ignore_index: int
    label_smoothing: float

    def __init__(
        self,
        stage: str,
        weight: torch.Tensor = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.stage = stage
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass masking stages"""

        probs = F.softmax(input, dim=-1)
        output = torch.argmax(probs, dim=-1)

        input_mask = output == PHENOLOGY_INT[self.stage]
        target_mask = target == PHENOLOGY_INT[self.stage]
        mask = input_mask | target_mask

        masked_loss = (
            F.cross_entropy(
                input,
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )
            * mask.float()
        )

        return masked_loss


class QuantileLoss(nn.Module):

    __constants__ = ["reduction"]

    def __init__(self, quantiles: list, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.quantiles = quantiles

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Quantile loss"""
        losses = []
        input = input.reshape(target.shape[0], len(self.quantiles), target.shape[1], target.shape[2])
        for i, q in enumerate(self.quantiles):
            errors = target - input[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)
        return losses


class ScaledMSELoss(nn.Module):
    """
    Scaled MSE Loss function, weighting later losses more
    """

    __constants__ = ["reduction"]

    def __init__(self, weight: int, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mask losses based on only current stage loss"""

        scale = self.weight * torch.arange(0, self.weight, self.weight / input.shape[1], device=input.device).unsqueeze(
            0
        )
        return F.mse_loss(input, target, reduction=self.reduction) * scale
