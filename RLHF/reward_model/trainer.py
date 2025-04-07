# trainer.py
import torch
import numpy as np
from transformers import Trainer
from typing import Dict, Union, Tuple, Optional, Any


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes binary classification accuracy and mean squared error (MSE)
    for reward model evaluation.

    Args:
        eval_pred (Tuple): A tuple of (predictions, labels) from the evaluation loop.

    Returns:
        dict: Dictionary containing 'accuracy' and 'mse'.
    """
    predictions, labels = eval_pred

    if predictions.ndim == 3:
        predictions = predictions.squeeze(-1)
    if predictions.ndim == 2:
        predictions = predictions[np.arange(predictions.shape[0]), -1]

    labels = labels.flatten()

    preds_binary = (predictions >= 0.5).astype(int)
    labels_binary = (labels >= 0.5).astype(int)

    accuracy = (preds_binary == labels_binary).mean()
    mse = ((predictions - labels) ** 2).mean()

    return {
        "accuracy": accuracy,
        "mse": mse,
    }


class RewardTrainer(Trainer):
    """
    Custom Trainer for training reward models using MSE loss.

    Overrides compute_loss and prediction_step for task-specific logic.
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Computes MSE loss between the final token value prediction and the label.

        Args:
            model: The model being trained.
            inputs: Dict containing input_ids, attention_mask, and labels.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss or (loss, outputs) depending on return_outputs.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"].float()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        values = outputs[2]  # Predicted scalar values (value head)

        # Get final token's value prediction
        seq_lens = attention_mask.sum(dim=1) - 1
        last_values = values.squeeze(-1)[
            torch.arange(values.size(0), device=values.device), seq_lens
        ]

        loss = torch.nn.functional.mse_loss(last_values, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[Any] = None,
    ) -> Tuple[None, torch.Tensor, torch.Tensor]:
        """
        Custom prediction step that extracts scalar values and ground-truth labels.

        Args:
            model: The model used for evaluation.
            inputs: Batch of inputs including labels.
            prediction_loss_only: Not used (required by interface).
            ignore_keys: Not used (required by interface).

        Returns:
            A tuple of (None, predictions, labels) with values on CPU.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"].float()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            values = outputs[2]
            seq_lens = attention_mask.sum(dim=1) - 1
            last_values = values.squeeze(-1)[
                torch.arange(values.size(0), device=values.device), seq_lens
            ]

        return None, last_values.detach().cpu(), labels.detach().cpu()
