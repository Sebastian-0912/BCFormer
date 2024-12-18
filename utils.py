import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import CLIPModel,AutoTokenizer,AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute mAP, CP, CR, CF1, OP, OR, OF1 for multi-label classification.

    Args:
        predictions (torch.Tensor): Model predictions of shape [batch_size, num_classes].
        targets (torch.Tensor): Ground truth labels of shape [batch_size, num_classes].
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        dict: Dictionary containing mAP, CP, CR, CF1, OP, OR, OF1.
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Convert probabilities to binary predictions
    binary_predictions = (predictions > threshold).astype(int)

    # --- mAP ---
    ap_per_class = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:  # Avoid classes with no positives
            ap = average_precision_score(targets[:, i], predictions[:, i])
            ap_per_class.append(ap)
    mAP = np.mean(ap_per_class) if ap_per_class else 0.0

    # --- CP, CR, CF1 (Class-wise Precision, Recall, F1) ---
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, binary_predictions, average=None, zero_division=0
    )
    CP = np.mean(precision)
    CR = np.mean(recall)
    CF1 = np.mean(f1)

    # --- OP, OR, OF1 (Overall Precision, Recall, F1) ---
    # Flatten to compute overall metrics
    total_true_positives = np.sum((binary_predictions == 1) & (targets == 1))
    total_false_positives = np.sum((binary_predictions == 1) & (targets == 0))
    total_false_negatives = np.sum((binary_predictions == 0) & (targets == 1))

    OP = total_true_positives / (total_true_positives + total_false_positives + 1e-10)
    OR = total_true_positives / (total_true_positives + total_false_negatives + 1e-10)
    OF1 = 2 * OP * OR / (OP + OR + 1e-10)

    return {
        "mAP": mAP,
        "CP": CP,
        "CR": CR,
        "CF1": CF1,
        "OP": OP,
        "OR": OR,
        "OF1": OF1,
    }


# Helper Functions
def freeze_model_params(model):
    """
    Freeze the parameters of a model to prevent updates during training.
    """
    for param in model.parameters():
        param.requires_grad = False
        
def plot_loss_curves(train_losses, val_losses):
    """
    Plot the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig("bfc_loss_curve.png")
    plt.show()