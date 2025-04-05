import torch
import numpy as np
from transformers import Trainer

def compute_metrics(eval_pred):
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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"].float()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        values = outputs[2]

        seq_lens = attention_mask.sum(dim=1) - 1
        last_values = values.squeeze(-1)[torch.arange(values.size(0), device=values.device), seq_lens]

        loss = torch.nn.functional.mse_loss(last_values, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"].float()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            values = outputs[2]
            seq_lens = attention_mask.sum(dim=1) - 1
            last_values = values.squeeze(-1)[torch.arange(values.size(0), device=values.device), seq_lens]

        return None, last_values.detach().cpu(), labels.detach().cpu()
