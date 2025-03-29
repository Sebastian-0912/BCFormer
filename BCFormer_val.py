import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import DataLoader
from transformers import CLIPModel,AutoTokenizer,AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from dataloader import MultiLabelImageDataset
from utils import freeze_model_params, compute_metrics
from model.BFC_MLP import BFC_MLP
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_blip_model():
    """
    Load and prepare the BLIP model and processor.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()
    processor.image_processor.do_rescale = False  # prevent rescale twice  
    freeze_model_params(model)
    return processor, model

def load_clip_model():
    """
    Load and prepare the CLIP model and tokenizer.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor.feature_extractor.do_rescale = False  # prevent rescale twice  
    model.eval()
    freeze_model_params(model)
    return processor,tokenizer, model

def get_clip_text_embedding(captions, tokenizer, clip_model):
    """
    Get text embeddings from CLIP model for a list of captions.
    """
    inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**inputs)
    return text_embeddings

def get_clip_image_embedding(images, clip_processor, clip_model):
    """
    Get image embeddings from CLIP model for a batch of images.
    Args:
        images (torch.Tensor): Batch of input images.
        clip_processor: CLIPProcessor for preprocessing images.
        clip_model: CLIP model for generating image embeddings.
    Returns:
        torch.Tensor: Image embeddings from CLIP model.
    """
    with torch.no_grad():
        # Preprocess images using the CLIPProcessor
        processed_images = clip_processor(images=images, return_tensors="pt").to(device)
        image_embeddings = clip_model.get_image_features(**processed_images)
    return image_embeddings

# def compute_metrics(predictions, targets, threshold=0.5):
#     """
#     Compute mAP, CP, CR, CF1, OP, OR, OF1 for multi-label classification.

#     Args:
#         predictions (torch.Tensor): Model predictions of shape [batch_size, num_classes].
#         targets (torch.Tensor): Ground truth labels of shape [batch_size, num_classes].
#         threshold (float): Threshold to convert probabilities to binary predictions.

#     Returns:
#         dict: Dictionary containing mAP, CP, CR, CF1, OP, OR, OF1.
#     """
#     predictions = predictions.cpu().numpy()
#     targets = targets.cpu().numpy()

#     # Convert probabilities to binary predictions
#     binary_predictions = (predictions > threshold).astype(int)

#     # --- mAP ---
#     ap_per_class = []
#     for i in range(targets.shape[1]):
#         if targets[:, i].sum() > 0:  # Avoid classes with no positives
#             ap = average_precision_score(targets[:, i], predictions[:, i])
#             ap_per_class.append(ap)
#     mAP = np.mean(ap_per_class) if ap_per_class else 0.0

#     # --- CP, CR, CF1 (Class-wise Precision, Recall, F1) ---
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         targets, binary_predictions, average=None, zero_division=0
#     )
#     CP = np.mean(precision)
#     CR = np.mean(recall)
#     CF1 = np.mean(f1)

#     # --- OP, OR, OF1 (Overall Precision, Recall, F1) ---
#     # Flatten to compute overall metrics
#     total_true_positives = np.sum((binary_predictions == 1) & (targets == 1))
#     total_false_positives = np.sum((binary_predictions == 1) & (targets == 0))
#     total_false_negatives = np.sum((binary_predictions == 0) & (targets == 1))

#     OP = total_true_positives / (total_true_positives + total_false_positives + 1e-10)
#     OR = total_true_positives / (total_true_positives + total_false_negatives + 1e-10)
#     OF1 = 2 * OP * OR / (OP + OR + 1e-10)

#     return {
#         "mAP": mAP,
#         "CP": CP,
#         "CR": CR,
#         "CF1": CF1,
#         "OP": OP,
#         "OR": OR,
#         "OF1": OF1,
#     }

def evaluate(dataloader, smoke_model, blip_model, blip_processor, clip_model, clip_processor, clip_tokenizer):
    """
    Evaluate the model on the validation dataset and return the average loss.
    """
    smoke_model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, true_labels in dataloader:
            images = images.to(device)
            true_labels = true_labels.to(device)
            gray_image = images.mean(dim=1)
            fft_result = torch.fft.fft2(gray_image)
            fft_shifted = torch.fft.fftshift(fft_result)
            phase_info = torch.angle(fft_shifted)

            # Generate captions using BLIP
            blip_inputs = blip_processor(images, return_tensors="pt").to(device)
            generated_ids = blip_model.generate(**blip_inputs)
            captions = [blip_processor.decode(g, skip_special_tokens=True) for g in generated_ids]

            # Get CLIP text embeddings
            text_embeddings = get_clip_text_embedding(captions, clip_tokenizer, clip_model)
            image_embeddings = get_clip_image_embedding(images, clip_processor, clip_model)
            
            combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)

            # Forward pass through MLP
            outputs = smoke_model(phase_info, combined_embeddings)
            
            # Collect predictions and targets for metric computation
            all_predictions.append(outputs.sigmoid())  # Use sigmoid to get probabilities
            all_targets.append(true_labels)

    # Stack predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)

    return metrics

# Main Training Loop
def main():
    # Load BLIP and CLIP
    blip_processor, blip_model = load_blip_model()
    clip_processor, clip_tokenizer, clip_model = load_clip_model()

    # Dataset and Dataloader
    images_dir = "./train_val_dataset/val"
    labels_file = "./train_val_dataset/val_labels.txt"
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    val_dataset = MultiLabelImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize Model
    crop_size = 16
    hidden_dim = 256
    num_classes = 3
    smoke_model = BFC_MLP(crop_size, hidden_dim, num_classes).to(device)
    smoke_model.load_state_dict(torch.load("./model_pth/bfc_ep10.pth"))
    # Evaluate
    metrics = evaluate(val_dataloader, smoke_model, blip_model, blip_processor, clip_model, clip_processor, clip_tokenizer)
    
    print(f"Metrics:")
    print(f"  mAP:  {metrics['mAP']:.4f}")
    print(f"  CP:   {metrics['CP']:.4f}")
    print(f"  CR:   {metrics['CR']:.4f}")
    print(f"  CF1:  {metrics['CF1']:.4f}")
    print(f"  OP:   {metrics['OP']:.4f}")
    print(f"  OR:   {metrics['OR']:.4f}")
    print(f"  OF1:  {metrics['OF1']:.4f}")

if __name__ == "__main__":
    main()